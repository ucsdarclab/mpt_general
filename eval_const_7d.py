''' A script to test the constraint planning model for panda
'''

from torch.nn import functional as F

import time
import skimage.io
from os import path as osp
from scipy import stats
from functools import partial
from torch.distributions import MultivariateNormal

import numpy as np
import torch
import torch.autograd as tag
import json
import argparse
import pickle
import open3d as o3d
import torch_geometric.data as tg_data
import pybullet as pyb
import roboticstoolbox as rtb

import matplotlib.pyplot as plt
from einops import rearrange

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    raise "Run code from a container with OMPL installed"

import scipy.optimize as opt

from modules.quantizers import VectorQuantizer
from modules.decoder import DecoderPreNorm, DecoderPreNormGeneral
from modules.encoder import EncoderPreNorm

from modules.autoregressive import AutoRegressiveModel, EnvContextCrossAttModel

from panda_utils import get_pybullet_server, set_env, q_max, q_min, set_robot, set_simulation_env
from panda_utils import ValidityCheckerDistance
import panda_utils as pu
from panda_shelf_env import place_shelf_and_obstacles
from ompl_utils import get_ompl_state, get_numpy_state

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

class StateSamplerRegion(ob.StateSampler):
    '''A class to sample robot joints from a given joint configuration.
    '''
    def __init__(self, space, qMin=None, qMax=None, dist_mu=None, dist_sigma=None):
        '''
        :param space:
        :param qMin: np.array of minimum joint bound
        :param qMax: np.array of maximum joint bound
        :param region: np.array of points to sample from
        '''
        super(StateSamplerRegion, self).__init__(space)
        self.name_ ='region'
        self.q_min = qMin
        self.q_max = qMax
        if dist_mu is None:
            self.X = None
            self.U = stats.uniform(np.zeros_like(qMin), np.ones_like(qMax))
        else:
            # self.X = MultivariateNormal(dist_mu,torch.diag_embed(dist_sigma))
            self.seq_num = dist_mu.shape[0]
            self.X = MultivariateNormal(dist_mu, dist_sigma)

                       
    def get_random_samples(self):
        '''Generates a random sample from the list of points
        '''
        index = 0
        random_samples = np.random.permutation(self.X.sample()*(self.q_max-self.q_min)+self.q_min)

        while True:
            yield random_samples[index, :]
            index += 1
            if index==self.seq_num:
                random_samples = np.random.permutation(self.X.sample()*(self.q_max-self.q_min)+self.q_min)
                index = 0
                
    def sampleUniform(self, state):
        '''Generate a sample from uniform distribution or key-points
        :param state: ompl.base.Space object
        '''
        if self.X is None:
            sample_pos = ((self.q_max-self.q_min)*self.U.rvs()+self.q_min)[0]
        else:
            sample_pos = next(self.get_random_samples())
        for i, val in enumerate(sample_pos):
            state[i] = val
        return True


def getPathLengthObjective(cost, si):
    '''
    Return the threshold objective for early termination
    :param cost: The cost of the original RRT* path
    :param si: An object of class ob.SpaceInformation
    :returns : An object of class ob.PathLengthOptimizationObjective
    '''
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostThreshold(ob.Cost(cost))
    return obj

import panda_constraint_shelf as pcs
import roboticstoolbox as rtb

def get_constraint_path(start, goal, env_num, dist_mu=None, dist_sigma=None, planner_type='rrtstar', **kwargs):
    '''
    Plan a path given the start, goal and patch_map.
    :param start:
    :param goal:
    :param goal_ori:
    :param env_num:
    :param dist_mu:
    :param dist_sigma:
    :param cost:
    :param planner_type:
    returns (list, float, int, bool): Returns True if a path was planned successfully.
    '''
    p = get_pybullet_server('direct')
    # Env - Shelf
    set_simulation_env(p)
    pandaID, jointsID, _ = set_robot(p)
    all_obstacles = place_shelf_and_obstacles(p, seed=env_num) 

    # Planning parameters
    space = ob.RealVectorStateSpace(7)
    bounds = ob.RealVectorBounds(7)
    # Set joint limits
    for i in range(7):
        bounds.setHigh(i, q_max[0, i])
        bounds.setLow(i, q_min[0, i])
    space.setBounds(bounds)
    pu.set_position(pandaID, jointsID, goal)
    _, goal_ori = pcs.get_robot_pose_orientation(pandaID)

    # # Redo the state sampler
    state_sampler = partial(StateSamplerRegion, dist_mu=dist_mu, dist_sigma=dist_sigma, qMin=q_min, qMax=q_max)
    space.setStateSamplerAllocator(ob.StateSamplerAllocator(state_sampler))
    
    # Define constraint function
    # tolerance = np.array([0.1, 2*np.pi, 2*np.pi])
    # tolerance = np.array([0.1, 0.1, 2*np.pi])
    tolerance = np.array([2*np.pi, 0.1, 0.1])
    # constraint_function = pcs.EndEffectorConstraint(goal_ori, tolerance, pandaID, jointsID)
    # TODO: integrate this w/ the end-effector constraint.
    panda_model = rtb.models.DH.Panda()
    fix_orient_R = panda_model.fkine(start).R
    constraint_function = pcs.EndEffectorConstraint(fix_orient_R, tolerance, pandaID, jointsID)
    
    # Set up the constraint planning space.
    css = ob.ProjectedStateSpace(space, constraint_function)
    csi = ob.ConstrainedSpaceInformation(css)
    
    # Define validity checker
    ss = og.SimpleSetup(csi)
    validity_checker_obj = ValidityCheckerDistance(csi, all_obstacles, pandaID, jointsID)
    ss.setStateValidityChecker(validity_checker_obj)

    # Define the start and goal state
    start_state = ob.State(csi.getStateSpace())
    goal_state = ob.State(csi.getStateSpace())

    for i in range(7):
        start_state[i] = start[i]
        goal_state[i] = goal[i]

    success = False

    # Define planning problem
    ss.setStartAndGoalStates(start_state, goal_state)

    if planner_type=='rrtstar':
        planner = og.RRTstar(csi)
    elif planner_type=='informedrrtstar':
        planner = og.InformedRRTstar(csi)
    elif planner_type=='bitstar':
        planner = og.BITstar(csi)
        planner.setSamplesPerBatch(500)
        planner.setPruneThresholdFraction(0.01)
        planner.setUseKNearest(False)
    elif planner_type=='rrtconnect':
        planner = og.RRTConnect(csi)
        # planner.setRange(13)
    else:
        print("None of the planners found, using RRT")
        planner = og.RRT(csi)
        planner_type = 'rrt'
        planner.setRange(13.07)
        raise TypeError(f"Planner Type {planner_type} not found")

    # planner.setRange(0.1)
    # Set the problem instance the planner has to solve

    ss.setPlanner(planner)
    ss.setup()

    # Attempt to solve the planning problem in the given time
    start_time = time.time()
    time_to_plan = kwargs['plan_time'] if 'plan_time' in kwargs.keys() else 300.0
    solved = planner.solve(time_to_plan)
    # if not pdef.hasExactSolution():
    #     # Redo the state sampler
    #     state_sampler = partial(StateSamplerRegion, dist_mu=None, dist_sigma=None, qMin=q_min, qMax=q_max)
    #     space.setStateSamplerAllocator(ob.StateSamplerAllocator(state_sampler))
    #     solved = planner.solve(25.0)
    plan_time = time.time()-start_time
    plannerData = ob.PlannerData(csi)
    planner.getPlannerData(plannerData)
    numVertices = plannerData.numVertices()

    if ss.haveExactSolutionPath():
        success = True
        # Simplify solution
        ss.simplifySolution()
        ss.getSolutionPath().interpolate()
        path = ss.getSolutionPath().printAsMatrix()
        # Convert path to a numpy array
        path = np.array(list(map(lambda x: np.fromstring(x, dtype=np.float32, sep=' '), path.split('\n')))[:-2])
        print(f"Path length after path simplification: {ss.getSolutionPath().length()}")
    else:
        path = [start, goal]
    
    return path, plan_time, numVertices, success


def get_beam_search_path(max_length, K, context_output, ar_model, quantizer_model, goal_index):
    ''' A beam search function, that stops when any of the paths hits termination.
    :param max_length: Max length to search.
    :param K: Number of paths to keep.
    :param context_output: the tensor ecoding environment information.
    :param ar_model: nn.Model type for the Auto-Regressor.
    :param quantizer_model: For extracting the feature vector.
    :param goal_index: Index used to mark end of sequence
    '''
    
    # Create place holder for input sequences.`
    input_seq = torch.ones(K, max_length, 512, dtype=torch.float, device=device)*-1
    quant_keys = torch.ones(K, max_length)*-1
    mask = torch.zeros(K, max_length+2, device=device)
        
    ar_model_input_i = torch.cat([context_output.repeat((K ,1, 1)), input_seq], dim=1)
    # mask the start/goal encoding and the prev. sequences.
    mask[:, :3] = 1

    # Get first set of quant_keys
    ar_output = ar_model(ar_model_input_i, mask)
    intial_cost = F.log_softmax(ar_output[:, 2, :], dim=-1)
    # Do not terminate on the final dictionary
    intial_cost[:, goal_index] = -1e9
    path_cost, start_index = intial_cost.topk(k=K, dim=-1)
    start_index = start_index[0]
    path_cost = path_cost[0]
    input_seq[:, 1, :] = quantizer_model.output_linear_map(quantizer_model.embedding(start_index))
    quant_keys[:, 0] = start_index
    for i in range(1, max_length-1):
        ar_model_input_i = torch.cat([context_output.repeat((K ,1, 1)), input_seq], dim=1)
        # mask the start/goal encoding and the prev. sequences.
        mask[:, :3+i] = 1
    
        ar_output = ar_model(ar_model_input_i, mask)
        
        # Get the sequence cost for the next step
        seq_cost = F.softmax(ar_output[:, 2+i, :], dim=-1)
        # Make self-loops impossible by setting the cost really low
        seq_cost[:, quant_keys[:, i-1].to(dtype=torch.int64)] = -1e9

        # Get the top set of possible sequences by flattening across batch sizes.
        nxt_cost, flatten_index = (path_cost[:, None]+seq_cost).flatten().topk(K)
        # Reshape back into tensor size to get the approriate batch index and word index.
        new_sequence = torch.as_tensor(np.array(np.unravel_index(flatten_index.cpu().numpy(), seq_cost.shape)).T)

        # Update previous keys given the current prediction.
        quant_keys[:, :i] = quant_keys[new_sequence[:, 0], :i]
        # Update the current set of keys.
        quant_keys[:, i] = new_sequence[:, 1].to(dtype=torch.float)
        # Update the cost
        path_cost = nxt_cost

        # Break at the first sign of termination
        if (new_sequence[:, 1] == goal_index).any():
            break

        # Select index
        select_index = new_sequence[:, 1] != goal_index

        # Update the input embedding. 
        input_seq[select_index, :i+1, :] = input_seq[new_sequence[select_index, 0], :i+1, :]
        input_seq[select_index, i+1, :] = quantizer_model.output_linear_map(quantizer_model.embedding(new_sequence[select_index, 1].to(device)))
    return quant_keys, path_cost, input_seq


def get_search_dist(normalized_path, path, map_data, context_encoder, decoder_model, ar_model, quantizer_model, num_keys):
    '''
    :returns (torch.tensor, torch.tensor, float): Returns an array of mean and covariance matrix and the time it took to 
    fetch them.
    '''
    # Get the context.
    start_time = time.time()
    start_n_goal = torch.as_tensor(normalized_path[[0, -1], :7], dtype=torch.float)
    env_input = tg_data.Batch.from_data_list([map_data])
    context_output = context_encoder(env_input, start_n_goal[None, :].to(device))
    # Find the sequence of dict values using beam search
    goal_index = num_keys+1
    quant_keys, _, input_seq = get_beam_search_path(51, 3, context_output, ar_model, quantizer_model, goal_index)
    # # -------------------------------- Testing training data ------------------------------
    # # Load quant keys from data.
    # with open(osp.join(args.dict_model_folder, 'quant_key', 'pandav3', 'val', f'env_{env_num:06d}', f'path_{path_num}.p'), 'rb') as f:
    #     quant_data = pickle.load(f)
    #     input_seq = quantizer_model.output_linear_map(quantizer_model.embedding(torch.tensor(quant_data['keys'], device=device)[None, :]))
    #     quant_keys = torch.cat((torch.tensor(quant_data['keys']), torch.tensor([goal_index])))[None, :]
    # # --------------------------------- x ------------- x ---------------------------------

    reached_goal = torch.stack(torch.where(quant_keys==goal_index), dim=1)
    if len(reached_goal) > 0:
        # Get the distribution.
        # Ignore the zero index, since it is encoding representation of start vector.
        output_dist_mu, output_dist_sigma = decoder_model(input_seq[reached_goal[0, 0], 1:reached_goal[0, 1]+1][None, :])
        # # -------------------------------- Testing training data ------------------------------
        # output_dist_mu, output_dist_sigma = decoder_model(input_seq)
        # # --------------------------------- x ------------- x ---------------------------------
        dist_mu = output_dist_mu.detach().cpu()
        dist_sigma = output_dist_sigma.detach().cpu()
        # If only a single point is predicted, then reshape the vector to a 2D tensor.
        if len(dist_mu.shape) == 1:
            dist_mu = dist_mu[None, :]
            dist_sigma = dist_sigma[None, :]
        # NOTE: set the 7th joint to zero degrees.
        # # ========================== No added goal =================================
        # search_dist_mu = torch.zeros((reached_goal[0, 1], 7))
        # search_dist_mu[:, :6] = output_dist_mu
        # # search_dist_sigma = torch.ones((reached_goal[0, 1], 7))
        # # search_dist_sigma[:, :6] = output_dist_sigma
        # search_dist_sigma = torch.diag_embed(torch.ones((reached_goal[0, 1], 7)))
        # search_dist_sigma[:, :6, :6] = torch.tensor(output_dist_sigma)
        # # ==========================================================================
        # ========================== append search with goal  ======================
        search_dist_mu = torch.zeros((reached_goal[0, 1]+1, 7))
        search_dist_mu[:reached_goal[0, 1], :7] = dist_mu
        search_dist_mu[reached_goal[0, 1], :] = torch.tensor(normalized_path[-1])
        search_dist_sigma = torch.diag_embed(torch.ones((reached_goal[0, 1]+1, 7)))
        search_dist_sigma[:reached_goal[0, 1], :7, :7] = torch.tensor(dist_sigma)
        search_dist_sigma[reached_goal[0, 1], :, :] = search_dist_sigma[reached_goal[0, 1], :, :]*0.01
        # ==========================================================================
    else:
        search_dist_mu = None
        search_dist_sigma = None
    patch_time = time.time()-start_time
    return search_dist_mu, search_dist_sigma, patch_time

import spatialmath as spm

def calculate_angle_error(delta_orient):
    '''
    Convert error in angle orientation to angular error.
    :param delta_orient: Change in orientation in rotation matrix.
    :returns np.array: error written in angular error.
    '''
    angle, axis = spm.base.tr2angvec(delta_orient)
    return angle*axis

class ConstraintFunctions:
    ''' A helper class for calculating constraints'''
    def __init__(self, initial_joint_pose, tolerance):
        '''
        :param client_id:
        :param quantizer_model: 
        :param initial_joint_pose: The initial joint config, used to set orientation constraint.
        :param tolerance: The tolerance of each constraint.
        '''
        self.tolerance = tolerance
        self.panda_model = rtb.models.DH.Panda()
        self.target_ori = self.panda_model.fkine(initial_joint_pose).R

    def function(self, x):
        '''
        Evaluate the value of the function at the given point x
        and set its value to out.
        :param x: value at state.
        :returns np.array: Objective function.
        '''
        orient_diff = self.target_ori.T@self.panda_model.fkine(x).R
        angle, axis = spm.base.tr2angvec(orient_diff)
        axis_error = angle*axis
        out = np.zeros_like(axis_error)
        for i in range(3):
            if  abs(axis_error[i])<self.tolerance[i]:
                out[i] = 0.0
            else:
                out[i] = abs(axis_error[i])
        return out

    def function_batch(self, x):
        '''
        Evaluate the function for a batch of data.
        '''
        # Get the pose of the robot.
        orient_diff = self.target_ori.T[None, :, :]@self.panda_model.fkine(x).R
        F = np.array(list(map(calculate_angle_error, orient_diff)))
        for i in range(3):
            sel_index = np.abs(F[:, i])<self.tolerance[i]
            F[:, i] = np.abs(F[:, i])
            F[sel_index, i] = 0.0
        return F

    def bound_derviate(self, error):
        '''
        Returns the jacobian of the bound.
        '''
        gradient = np.zeros((3, 3))
        for i, e_i in enumerate(error):
            if abs(e_i)>self.tolerance[i]:
                gradient[i, i] = e_i/abs(e_i)
        return gradient

    def jacobian(self, q):
        '''
        Evaluate the Jacobian of the objective function with respect to joint angles.
        :param q: value at state.
        :returns np.array: numpy array of joint_dim X 3
        '''
        orient_diff = self.target_ori.T@self.panda_model.fkine(q).R
        angle, axis = spm.base.tr2angvec(orient_diff)
        bound_gradient = self.bound_derviate(angle*axis)
        error_jacobian = bound_gradient@self.target_ori.T@self.panda_model.jacob0(q, half='rot')
        return error_jacobian

    def jacobian_batch(self, q):
        '''
        Evaluate the Jacobian of the F(q)^TF(q) wrt to the joint angles for a batch of angles.
        :param q: An array of joint angles.
        :returns np.array: numpy array of batch jacobians.
        '''
        orient_diff = self.target_ori.T[None, :, :]@self.panda_model.fkine(q).R
        F = np.array(list(map(calculate_angle_error, orient_diff)))
        bound_gradient = np.zeros((q.shape[0], 3, 3))
        for i in range(3):
            sel_index = np.abs(F[:, i])<self.tolerance[i]
            bound_gradient[:, i, i] = F[:, i]/np.abs(F[:, i])
            bound_gradient[sel_index, i] = 0.0
        robot_jacobian = np.array(list(map(lambda q_i: self.panda_model.jacob0(q_i, half='rot'), q)))
        error_jacobian = bound_gradient@self.target_ori.T[None, :]@robot_jacobian
        return error_jacobian 


def back_tracking(f, x0, delta, gamma, N):
    '''
    Do the line search along the function f.
    :param f: Function to do backtracking.
    :param x0: Initial value of search.
    :param delta: The direction to move.
    :param gamma: The fraction to backtrack.
    :param N: Maximum number of iterations.
    '''
    f0 = f(x0)
    x = x0
    alpha = 1
    for i in range(N):
        x = x - alpha*delta
        fx = f(x)
        print(fx, alpha)
        if np.linalg.norm(fx)<0.1 or (np.linalg.norm(fx)/np.linalg.norm(f0))<1:
            return x
        alpha = gamma*alpha
    return None

def get_search_proj_dist(normalized_path, path, map_data, context_encoder, decoder_model, ar_model, quantizer_model, num_keys):
    '''
    :returns (torch.tensor, torch.tensor, float): Returns an array of mean and covariance matrix and the time it took to 
    fetch them.
    '''
    # Get the context.
    # tolerance = np.array([0.1, 2*np.pi, 2*np.pi])
    # tolerance = np.array([0.1, 0.1, 2*np.pi])
    tolerance = np.array([2*np.pi, 0.1, 0.1])
    constraint_obj = ConstraintFunctions(path[0], tolerance = tolerance)
    start_time = time.time()
    start_n_goal = torch.as_tensor(normalized_path[[0, -1], :7], dtype=torch.float)
    env_input = tg_data.Batch.from_data_list([map_data])
    context_output = context_encoder(env_input, start_n_goal[None, :].to(device))
    # Find the sequence of dict values using beam search
    goal_index = num_keys+1
    quant_keys, _, input_seq = get_beam_search_path(51, 3, context_output, ar_model, quantizer_model, goal_index)

    reached_goal = torch.stack(torch.where(quant_keys==goal_index), dim=1)
    if len(reached_goal) > 0:
        scale = torch.diag(torch.tensor((pu.q_max[0]-pu.q_min[0]), device=device, dtype=torch.float))
        qMin_tensor = torch.tensor(pu.q_min[0], device=device, dtype=torch.float)
        def get_distribution_mean(z):
            '''
            Returns the distribution for the latent vector z.
            :param z: The latent vector z.
            :returns torch.tensor: The mean distribution of z.
            '''
            z_scaled = quantizer_model.output_linear_map(z)
            dist_mu, _ = decoder_model(z_scaled[None, :])
            dist_mu = qMin_tensor + scale@dist_mu.squeeze()
            return dist_mu
        # Get the distribution.
        # Ignore the zero index, since it is encoding representation of start vector.
        # TODO: This is normalized space !!!
        output_dist_mu, _ = decoder_model(input_seq[reached_goal[0, 0], 1:reached_goal[0, 1]+1][None, :])
        output_dist_mu = output_dist_mu.detach().squeeze().cpu().numpy()
        if len(output_dist_mu.shape)==1:
            output_dist_mu = output_dist_mu[None, :]
        # Modify the latent vector by projecting along the gradient.
        z_latent = quantizer_model.embedding(quant_keys[reached_goal[0, 0], :reached_goal[0, 1]].to(dtype=torch.int, device=device))
        z_latent_new = torch.zeros_like(z_latent)
        # Get new latent varialbes using optimization
        def reg_constrain_function(z_i, q_c, gamma):
            '''
            Calculates the regularized constraint objective function for a given
            latent variable.
            :param z_i: The given latent variable
            :param target_ori: The constraint orientation.
            :param q_c: Initial joint configuration.
            :param gamma: weight factor for regularization term.
            :param tolerance: tolerance for the constraint.
            :returns float: a scalar value.
            '''
            z_i = torch.tensor(z_i, device=device, dtype=torch.float)
            q_i = get_distribution_mean(z_i).detach().cpu().numpy()
            f = constrain_obj.function(q_i)
            g = (q_i-q_c).T@(q_i-q_c)
            return f.T@f + gamma*g
        def reg_constrain_jacobian(z_i, q_c, gamma):
            '''
            Returns the Jacobian of the regularized constraint objective function for
            the given latent variable.
            :param z_i: latent variable.
            :param target_ori: The constraint orientation.
            :param q_c: Initial joint configuration.
            :param gamma: weight factor for the regularization term.
            :param tolerance: tolerance used for the constraint.
            :return np.arrray: jacobian of the objective fuction.
            '''
            z_i = torch.tensor(z_i, device=device, dtype=torch.float)
            q_i = get_distribution_mean(z_i).detach().cpu().numpy()
            f = constrain_obj.function(q_i)
            J_const = constrain_obj.jacobian(q_i)
            J_decoder = tag.functional.jacobian(get_distribution_mean, z_i).detach().cpu().numpy()
            J = 2*(f.T@J_const + gamma*(q_i-q_c).T)@J_decoder
            return J
        gamma = 0.1
        z_latent_new = torch.zeros_like(z_latent)
        for i, z_i in enumerate(z_latent):
            q_i = get_distribution_mean(z_i).detach().cpu().numpy()
            start_cost = constrain_obj.function(q_i)
            objective_fun = lambda z: reg_constrain_function(z, q_i, gamma)
            objective_jac = lambda z: reg_constrain_jacobian(z, q_i, gamma)
            sol = opt.minimize(
                fun=objective_fun,
                x0=z_i.detach().cpu().numpy(),
                jac=objective_jac,
                method='SLSQP',
                options={'disp':False},
                # constraints={'type':'eq', 'fun':lambda x:x.T@x-1}
            )
            if sol.success:
                z_temp = torch.tensor(sol.x, device=device, dtype=torch.float)
                z_temp = F.normalize(z_temp[None, :]).squeeze()
            else:
                print("No solution found")
                z_temp = z_i
            z_latent_new[i, :] = z_temp
            new_cost = constrain_obj.function(get_distribution_mean(z_temp).detach().cpu().numpy())
            print(f"Cost(Before): {start_cost} \nCost(After):{new_cost}")
        # # Get new latent variables using line search.
        # def get_jacobian(z_i):
        #     q_i = get_distribution_mean(z_i).detach().cpu().numpy()
        #     f = constrain_obj.function(q_i)
        #     J_const = constrain_obj.jacobian(q_i)
        #     J_decoder = tag.functional.jacobian(get_distribution_mean, z_i)
        #     J = J_const@J_decoder.detach().cpu().numpy()
        #     return f, J
        # for i, z_i in enumerate(z_latent):
        #     # Run a Newton algorithm to find zero-crossings
        #     z0 = z_i
        #     for _ in range(20):
        #         f, J = get_jacobian(z0)
        #         print(f"Start Cost: {f}")
        #         # Terminate when we reach the line
        #         if np.linalg.norm(f)<=0.1:
        #             break
        #         delta_z = torch.tensor(np.linalg.pinv(J)@f, device=device, dtype=torch.float)
        #         # Use a backtracking algorithm:
        #         f_z = lambda x: constrain_obj.function(get_distribution_mean(x).detach().cpu().numpy())
        #         z_temp = back_tracking(f_z, z0, delta_z, gamma=0.9, N=10)
        #         if z_temp is not None:
        #             z0 = F.normalize(z_temp[None, :]).squeeze()
        #         else:
        #             break
        #         # import pdb;pdb.set_trace()

        #     z_latent_new[i, :] = z0
        # Get closest dictionary index.
        # d = -torch.einsum('bd, dn -> bn', z_latent_new, rearrange(quantizer_model.embedding.weight, 'n d -> d n'))
        # new_dict_code = torch.argmin(d, axis=1)
        # z_latent = quantizer_model.embedding(new_dict_code)
        # projected_latent = quantizer_model.output_linear_map(z_latent)
        projected_latent = quantizer_model.output_linear_map(z_latent_new)
        output_dist_mu, output_dist_sigma = decoder_model(projected_latent[None, :])

        dist_mu = output_dist_mu.detach().cpu()
        dist_sigma = output_dist_sigma.detach().cpu()
        # If only a single point is predicted, then reshape the vector to a 2D tensor.
        if len(dist_mu.shape) == 1:
            dist_mu = dist_mu[None, :]
            dist_sigma = dist_sigma[None, :]
        
        # ========================== append search with goal  ======================
        search_dist_mu = torch.zeros((reached_goal[0, 1]+1, 7))
        search_dist_mu[:reached_goal[0, 1], :7] = dist_mu
        search_dist_mu[reached_goal[0, 1], :] = torch.tensor(normalized_path[-1])
        search_dist_sigma = torch.diag_embed(torch.ones((reached_goal[0, 1]+1, 7)))
        search_dist_sigma[:reached_goal[0, 1], :7, :7] = torch.tensor(dist_sigma)
        search_dist_sigma[reached_goal[0, 1], :, :] = search_dist_sigma[reached_goal[0, 1], :, :]*0.01
        # ==========================================================================
    else:
        search_dist_mu = None
        search_dist_sigma = None
    patch_time = time.time()-start_time
    return search_dist_mu, search_dist_sigma, patch_time


def main(args):
    ''' Main running script.
    :parma args: An object of type argparse.ArgumentParser().parse_args()
    '''
    use_model = False if args.dict_model_folder is None else True
    if use_model:
        print("Using model")
        # ========================= Load trained model ===========================
        # Define the models
        d_model = 512
        #TODO: Get the number of keys from the saved data
        num_keys = 2048
        goal_index = num_keys + 1
        quantizer_model = VectorQuantizer(n_e=num_keys, e_dim=8, latent_dim=d_model)

        # Load quantizer model.
        dictionary_model_folder = args.dict_model_folder
        with open(osp.join(dictionary_model_folder, 'model_params.json'), 'r') as f:
            dictionary_model_params = json.load(f)

        encoder_model = EncoderPreNorm(**dictionary_model_params)
        decoder_model = DecoderPreNormGeneral(
            e_dim=dictionary_model_params['d_model'], 
            h_dim=dictionary_model_params['d_inner'], 
            c_space_dim=dictionary_model_params['c_space_dim']
        )

        checkpoint = torch.load(osp.join(dictionary_model_folder, 'best_model.pkl'))
        
        # Load model parameters and set it to eval
        for model, state_dict in zip([encoder_model, quantizer_model, decoder_model], ['encoder_state', 'quantizer_state', 'decoder_state']):
            model.load_state_dict(checkpoint[state_dict])
            model.eval()
            model.to(device)

        # Load the AR model.
        # NOTE: Save these values as dictionary in the future, and load as json.
        env_params = {
            'd_model': dictionary_model_params['d_model'],
        }
        ar_model_folder = args.ar_model_folder
        # Create the environment encoder object.
        with open(osp.join(ar_model_folder, 'cross_attn.json'), 'r') as f:
            context_env_encoder_params = json.load(f)
        context_env_encoder = EnvContextCrossAttModel(env_params, context_env_encoder_params, robot='6D')
        # Create the AR model
        with open(osp.join(ar_model_folder, 'ar_params.json'), 'r') as f:
            ar_params = json.load(f)
        ar_model = AutoRegressiveModel(**ar_params)

        # Load the parameters and set the model to eval
        checkpoint = torch.load(osp.join(ar_model_folder, 'best_model.pkl'))
        for model, state_dict in zip([context_env_encoder, ar_model], ['context_state', 'ar_model_state']):
            model.load_state_dict(checkpoint[state_dict])
            model.eval()
            model.to(device)
    # ============================= Run planning experiment ============================
    val_data_folder = args.val_data_folder
    start = args.start
    pathSuccess = []
    pathTime = []
    pathTimeOverhead = []
    pathVertices = []
    pathPlanned = []
    predict_seq_time = []
    for env_num in range(start, start+args.samples):
        print(env_num)
        if use_model:
            map_file = osp.join(val_data_folder, f'env_{env_num:06d}/map_{env_num}.ply')
            data_PC = o3d.io.read_point_cloud(map_file, format='ply')
            depth_points = np.array(data_PC.points)
            map_data = tg_data.Data(pos=torch.as_tensor(depth_points, dtype=torch.float, device=device))

        for path_num in range(args.num_paths):
            path_file = osp.join(val_data_folder, f'env_{env_num:06d}/path_{path_num}.p')
            data = pickle.load(open(path_file, 'rb'))
            path = (data['path']-q_min)/(q_max-q_min)
            if data['success']:
                if use_model:
                    if args.project:
                        search_dist_mu, search_dist_sigma, patch_time = get_search_proj_dist(path, data['path'], map_data, context_env_encoder, decoder_model, ar_model, quantizer_model, num_keys)
                    else:
                        search_dist_mu, search_dist_sigma, patch_time = get_search_dist(path, data['path'], map_data, context_env_encoder, decoder_model, ar_model, quantizer_model, num_keys)
                else:
                    print("Not using model, using uniform distribution")
                    search_dist_mu, search_dist_sigma, patch_time = None, None, 0.0
            
                planned_path, t, v, s = get_constraint_path(data['path'][0], data['path'][-1], env_num, search_dist_mu, search_dist_sigma, planner_type=args.planner_type)
                print(t)
                pathSuccess.append(s)
                pathTime.append(t)
                pathVertices.append(v)
                pathTimeOverhead.append(t)
                pathPlanned.append(np.array(planned_path))
                predict_seq_time.append(patch_time)
            else:
                pathSuccess.append(False)
                pathTime.append(0)
                pathVertices.append(0)
                pathTimeOverhead.append(0)
                pathPlanned.append([[]])
                predict_seq_time.append(0)
    
    pathData = {'Time':pathTime, 'Success':pathSuccess, 'Vertices':pathVertices, 'PlanTime':pathTimeOverhead, 'PredictTime': predict_seq_time, 'Path': pathPlanned}
    if use_model:
        if args.project:
            # fileName = osp.join(ar_model_folder, f'eval_val_const_proj_plan_ls_{args.planner_type}_tip_{start:06d}.p')
            # fileName = osp.join(ar_model_folder, f'eval_val_const_proj_plan_ls_nearest_{args.planner_type}_tip_{start:06d}.p')
            fileName = osp.join(ar_model_folder, f'eval_val_const_proj_plan_opt_v5_{args.planner_type}_tip_{start:06d}.p')
        else:
            fileName = osp.join(ar_model_folder, f'eval_val_const_plan_{args.planner_type}_tip_{start:06d}.p')
    else:
        fileName = f'/root/data/general_mpt_panda_7d/const_{args.planner_type}_tip_{start:06d}.p'
    pickle.dump(pathData, open(fileName, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_model_folder', help="Folder where dictionary model is stored")
    parser.add_argument('--ar_model_folder', help="Folder where AR model is stored")
    parser.add_argument('--val_data_folder', help="Folder where environment data is stored")
    parser.add_argument('--start', help="Env number to start", type=int)
    parser.add_argument('--samples', help="Number of samples to collect", type=int)
    parser.add_argument('--num_paths', help="Number of paths for each environment", type=int)
    parser.add_argument('--planner_type', help="Type of planner to use", choices=['rrtstar', 'rrt', 'rrtconnect', 'informedrrtstar', 'fmtstar', 'bitstar'])
    parser.add_argument('--project', help="Project using the constraint function", action='store_true')

    args = parser.parse_args()
    main(args)