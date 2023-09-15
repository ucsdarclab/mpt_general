''' Scratch pad for kitchen environment
'''

import pybullet as pyb
import numpy as np
import os
from os import path as osp
import pickle
import time
from functools import partial
from scipy.spatial.transform import Rotation as Rot

from scipy import stats

import panda_utils as pu
from pybullet_object_models import ycb_objects

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    raise "Run code from a container with OMPL installed"

from torch.distributions import MultivariateNormal
from ompl_utils import get_ompl_state, get_numpy_state

import open3d as o3d
import networkx as nx

# Contact-GraspNet imports!!
import sys
BASE_DIR = '/root/third_party/contact_graspnet'
sys.path.append(osp.join(BASE_DIR, 'contact_graspnet'))

# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# from contact_grasp_estimator import GraspEstimator
# import config_utils
# from visualization_utils import visualize_grasps
import panda_constraint_shelf as pcs
from tracikpy import TracIKSolver
import interactive_panda_kitchen as ipk
import eval_const_7d as ec7

# from interactive_panda_kitchen import *
import torch
import torch_geometric.data as tg_data
import json

# VQ-MPT model 
from modules.quantizers import VectorQuantizer
from modules.decoder import DecoderPreNormGeneral
from modules.encoder import EncoderPreNorm
from modules.autoregressive import AutoRegressiveModel, EnvContextCrossAttModel

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

def add_debug_point(client_id, pose):
    colors = np.zeros_like(pose)
    colors[:, 0] = 1.0
    obj_id = client_id.addUserDebugPoints(pose, colors, 25)
    return obj_id
import roboticstoolbox as rtb
import spatialmath as smp

class TSR(ob.GoalSampleableRegion):
    ''' A class to define Task-Space Regions
    '''
    def __init__(self, si, validity_checker, world_T_obj, offset_T_grasp, goal_tolerance, space_dim=7):
        ''' Define constructor for the goal region
        :param si: ompl.base.StateInformation object.
        :param validity_checker: 
        :param world_T_obj: Position of the object in world co-ordinates
        :param offset_T_grasp: Offset from the object to robot ee.
        :param goal_tolerance: range of orientation to sample from.
        :param space_dim: dimensin of the input space.
        '''
        super(TSR, self).__init__(si)
        # set threshold for isSatisfied
        self.setThreshold(0.001)
        # check if ik solver can do forward kinematics and IK 
        self.ik_solver = TracIKSolver(
            "assets/franka_panda/franka_panda.urdf",
            "panda_link0",
            "panda_hand",
            # "panda_rightfinger",
            timeout=0.1,
            solve_type='Distance'
        )
        self.validity_checker = validity_checker
        self.world_T_obj = world_T_obj
        self.offset_T_grasp = offset_T_grasp
        self.tolerance = goal_tolerance
        self.space_dim = space_dim
        
    def distanceGoal(self, state):
        ''' Return the distance from goal for the given state.
        '''
        # Take the FK of the current state
        np_state = np.array([state[i] for i in range(self.space_dim)])
        
        world_T_grasp = self.ik_solver.fk(np_state)
        world_T_offset = world_T_grasp@np.linalg.inv(self.offset_T_grasp)

        # Check the Eucledian distance b/w object pose and ee offset position.
        distance = np.linalg.norm(world_T_offset[:3, 3]-self.world_T_obj[:3, 3])
        if  distance>1e-1:
            return float(distance)
        
        # If distance is zero, check if orientation is w/ tolerance
        orient_diff = self.world_T_obj[:3, :3].T@world_T_offset[:3, :3]
        angle, axis = smp.base.tr2angvec(orient_diff)
        axis_error = abs(angle*axis)
        sel_index = axis_error<self.tolerance
        axis_error[sel_index] = 0.0
        return float(axis_error.T@axis_error)
    
    def sampleGoal(self, state):
        '''Samples a goal from within the goal region.
        :param state: 
        '''
        for _ in range(100):
            # Sample a ee-configuraion
            delta_orient = (np.random.rand(3)*2-1)*self.tolerance
            # Get the Axis angle form
            delta_theta = np.linalg.norm(delta_orient)
            delta_vector = delta_orient/delta_theta
            T_sample = smp.base.rt2tr(smp.base.angvec2r(delta_theta, delta_vector), np.zeros(3))
            world_T_sample = self.world_T_obj@T_sample

            # Add grasping offset for collision w/ obstacle.
            world_T_grasp = world_T_sample@self.offset_T_grasp

            # Check if a IK exists
            q_sample = self.ik_solver.ik(world_T_grasp)
            if q_sample is None:
                continue

            # check if it is in collison.
            if not self.validity_checker.isValid(q_sample):
                continue

            # If successful update state.
            for i in range(7):
                state[i] = float(q_sample[i])
            break
    
    # def isSatisfied(self, state, distance):
    #     distance =  self.distanceGoal(state)
    #     if np.isclose(distance, 0.0):
    #         return True
    #     return False

    def maxSampleCount(self):
        ''' Return the maximum number of samples that can be asked for before repeating.
        '''
        return 1000


def get_constraint_path(start, world_T_obj, validity_checker_obj, constraint_function_obj, dist_mu=None, dist_sigma=None, **kwargs):
    '''
    Plan a path given 
    '''
    # Planning parameters
    space = ob.RealVectorStateSpace(7)
    bounds = ob.RealVectorBounds(7)
    # Set joint limits
    for i in range(7):
        bounds.setHigh(i, pu.q_max[0, i])
        bounds.setLow(i, pu.q_min[0, i])
    space.setBounds(bounds)

    # Redo the state sampler
    state_sampler = partial(ipk.StateSamplerRegion, dist_mu=dist_mu, dist_sigma=dist_sigma, qMin=pu.q_min, qMax=pu.q_max)
    space.setStateSamplerAllocator(ob.StateSamplerAllocator(state_sampler))
    
    state_space = kwargs.get('state_space', 'PJ')
    # Set up the constraint planning space
    # PJ -> Projected
    # AT -> Atlas
    # TB -> Tangent Bundle
    if state_space == 'PJ':
        css = ob.ProjectedStateSpace(space, constraint_function_obj)
    elif state_space == 'AT':
        css = ob.AtlasStateSpace(space, constraint_function_obj)
    elif state_space == 'TB':
        css = ob.TangentBundleStateSpace(space, constraint_function_obj)
    else:
        raise ValueError(f"unkown state space : {state_space}")
    csi = ob.ConstrainedSpaceInformation(css)
    css.setup()
    # Parameters for Atlas and Tangent Bundle planners.
    # Find detail info abt parameters - https://ompl.kavrakilab.org/ConstrainedPlanningCommon_8py_source.html
    if not state_space == "PJ":
        css.setExploration(0.8)
        # css.setExploration(ob.ATLAS_STATE_SPACE_EXPLORATION)
        css.setEpsilon(0.1)
        # css.setEpsilon(ob.ATLAS_STATE_SPACE_EPSILON)
        css.setRho(ob.CONSTRAINED_STATE_SPACE_DELTA * ob.ATLAS_STATE_SPACE_RHO_MULTIPLIER)
        css.setAlpha(ob.ATLAS_STATE_SPACE_ALPHA)
        css.setMaxChartsPerExtension(ob.ATLAS_STATE_SPACE_MAX_CHARTS_PER_EXTENSION)
        # # Use frontier-based expansion
        # css.setBiasFunction(lambda c, atlas=css:atlas.getChartCount() - c.getNeighborCount() + 1.)
        # if state_space == "AT":
        #     css.setSeparated(not options.no_separate)
        css.setup()
    
    # Define validity checker
    ss = og.SimpleSetup(csi)
    ss.setStateValidityChecker(validity_checker_obj)

    # Define the start and goal state
    start_state = ob.State(csi.getStateSpace())
    for i in range(7):
        start_state[i] = start[i]

    success = False
    can_T_ee = np.array([[0., 0., 1, 0.], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    goal_region = TSR(
        csi, 
        validity_checker_obj, 
        world_T_obj, 
        offset_T_grasp=np.c_[np.eye(4, 3), np.array([-0.11, 0.0, 0.05, 1])]@can_T_ee, 
        goal_tolerance=np.array([0.0, 0.0, 0.5*np.pi])
    )
    # import pdb;pdb.set_trace()
    # Anchor start and goal states
    if not state_space == 'PJ':
        css.anchorChart(start_state())
        # TODO: Maybe anchor more goal states?
        sampled_goal_state = ob.State(csi.getStateSpace())
        # Add multiple goal regions as anchor points for the Atlas-Planner
        for _ in range(20):
            goal_region.sampleGoal(sampled_goal_state)
            css.anchorChart(sampled_goal_state())

    # Define start state and goal region using TSR representation
    ss.setStartState(start_state)
    ss.setGoal(goal_region)

    # Define planner
    planner = og.RRTConnect(csi)
    # planner.setRange(13.07)

    # Set the problem instance the planner has to solve
    ss.setPlanner(planner)
    ss.setup()

    # Attempt to solve the planning problem in the given time
    start_time = time.time()
    total_time = kwargs.get('plan_time', 30) # If not plan time found, use 30sec
    solved = planner.solve(total_time)
    # if not pdef.hasExactSolution():
    #     # Redo the state sampler
    #     state_sampler = partial(StateSamplerRegion, dist_mu=None, dist_sigma=None, qMin=q_min, qMax=q_max)
    #     space.setStateSamplerAllocator(ob.StateSamplerAllocator(state_sampler))
    #     solved = planner.solve(25.0)
    plan_time = time.time()-start_time
    planner_data = ob.PlannerData(csi)
    planner.getPlannerData(planner_data)
    numVertices = planner_data.numVertices()

    # G = nx.parse_graphml(planner_data.printGraphML())
    # all_pose = {node:np.array(list(map(float, G.nodes[node]['coords'].split(',')))) for node in G.nodes}
    # scale = torch.asarray(np.diag(pu.q_max[0]-pu.q_min[0]), dtype=torch.float)
    # scaled_dist_mu = dist_mu@scale+pu.q_min
    # fig, ax_all = plt.subplots(1, 3, figsize=(21, 7))
    # for i, ax in enumerate(ax_all):
    #     part = {node:value[2*i:2*(i+1)] for node,value in all_pose.items()}
    #     nx.draw_networkx(G, pos=part, ax=ax, with_labels=False, node_size=20, label='sampled', node_color='b')
    #     ax.scatter(*start[2*i:2*(i+1)], color='g', zorder=3)
    #     ax.scatter(*scaled_dist_mu[:, 2*i:2*(i+1)].T, color='r', zorder=3)
    #     ax.scatter(*kwargs['goal_samples'][:, 2*i:2*(i+1)].T, color='k', zorder=3)
    # fig.show()

    if ss.haveExactSolutionPath():
        print("Found solution")
        success = True
        # Simplify solution
        ss.simplifySolution()
        ss.getSolutionPath().interpolate()
        path = ss.getSolutionPath().printAsMatrix()
        # Convert path to a numpy array
        path = np.array(list(map(lambda x: np.fromstring(x, dtype=np.float32, sep=' '), path.split('\n')))[:-2])
        path_length = ss.getSolutionPath().length()
        print(f"Path length after path simplification: {ss.getSolutionPath().length()}")
    else:
        print("No solution")
        path = [start, world_T_obj]
        path_length = -1
    return path, path_length, plan_time, numVertices, success

def get_search_dist_batch(normalized_start, normalized_goals, map_data, context_encoder, decoder_model, ar_model, quantizer_model, num_keys):
    '''
    :returns (torch.tensor, torch.tensor, float): Returns an array of mean and covariance matrix and the time it took to 
    fetch them, given a batch of normalized goals.
    '''
    # Get the context.
    start_time = time.time()
    goal_index = num_keys+1
    # Get the dictionary variables
    all_quant_keys = []
    env_input = tg_data.Batch.from_data_list([map_data])
    for n_goal in normalized_goals:
        normalized_path = np.r_[normalized_start, n_goal[None, :]]
        start_n_goal = torch.as_tensor(normalized_path, dtype=torch.float)
        context_output = context_encoder(env_input, start_n_goal[None, :].to(device))
        # Find the sequence of dict values using beam search
        quant_keys, _, _ = ec7.get_beam_search_path(51, 3, context_output, ar_model, quantizer_model, goal_index)
        reached_goal = torch.stack(torch.where(quant_keys==goal_index), dim=1)
        if len(reached_goal)>0:
            all_quant_keys.append(quant_keys[reached_goal[0, 0], 1:reached_goal[0, 1]])

    all_quant_keys = torch.unique(torch.cat(all_quant_keys)).to(dtype=torch.int, device=device)

    # Convert quant keys to latent vectors
    latent_seq = quantizer_model.output_linear_map(quantizer_model.embedding(all_quant_keys))

    # Get the distribution.
    # Ignore the zero index, since it is encoding representation of start vector.
    output_dist_mu, output_dist_sigma = decoder_model(latent_seq[None, :])
    dist_mu = output_dist_mu.detach().cpu()
    dist_sigma = output_dist_sigma.detach().cpu()
    # If only a single point is predicted, then reshape the vector to a 2D tensor.
    if len(dist_mu.shape) == 1:
        dist_mu = dist_mu[None, :]
        dist_sigma = dist_sigma[None, :]
    
    # ========================== append search with goal  ======================
    num_distributions = all_quant_keys.shape[0]
    num_goals = normalized_goals.shape[0]
    search_dist_mu = torch.zeros((num_distributions+num_goals, 7))
    search_dist_mu[:num_distributions, :7] = dist_mu
    search_dist_mu[num_distributions:, :] = torch.tensor(normalized_goals)
    search_dist_sigma = torch.diag_embed(torch.ones((num_distributions+num_goals, 7)))
    search_dist_sigma[:num_distributions, :7, :7] = torch.tensor(dist_sigma)
    search_dist_sigma[num_distributions:, :, :] = search_dist_sigma[num_distributions:, :, :]*0.01
    # ==========================================================================
    
    patch_time = time.time()-start_time
    return search_dist_mu, search_dist_sigma, patch_time


def get_constraint_path_v2(start, goal, validity_checker_obj, constraint_function_obj, dist_mu=None, dist_sigma=None, **kwargs):
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

    # Planning parameters
    space = ob.RealVectorStateSpace(7)
    bounds = ob.RealVectorBounds(7)
    # Set joint limits
    for i in range(7):
        bounds.setHigh(i, pu.q_max[0, i])
        bounds.setLow(i, pu.q_min[0, i])
    space.setBounds(bounds)

    # # Redo the state sampler
    state_sampler = partial(ipk.StateSamplerRegion, dist_mu=dist_mu, dist_sigma=dist_sigma, qMin=pu.q_min, qMax=pu.q_max)
    space.setStateSamplerAllocator(ob.StateSamplerAllocator(state_sampler))
    
    # Set up the constraint planning space.
    css = ob.ProjectedStateSpace(space, constraint_function_obj)
    csi = ob.ConstrainedSpaceInformation(css)
    
    # Define validity checker
    ss = og.SimpleSetup(csi)
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
    planner = og.RRTConnect(csi)
    planner.setRange(13.07)

    # Set the problem instance the planner has to solve
    ss.setPlanner(planner)
    ss.setup()

    # Attempt to solve the planning problem in the given time
    start_time = time.time()
    time_to_plan = kwargs['plan_time'] if 'plan_time' in kwargs.keys() else 20
    solved = planner.solve(time_to_plan)
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
        path_length = ss.getSolutionPath().length()
        print(f"Path length after path simplification: {ss.getSolutionPath().length()}")
    else:
        path_length = -1
        path = [start, goal]
    
    return path, path_length, plan_time, numVertices, success

import matplotlib.pyplot as plt


if __name__=="__main__":
    # Server for collision checking
    p_collision = pu.get_pybullet_server('direct')
    # Server for collision checking
    p_pcd = pu.get_pybullet_server('direct')
    # Server for visualization
    p = pu.get_pybullet_server('gui')
    # p = pu.get_pybullet_server('direct')

    p.resetDebugVisualizerCamera(1.8, -28.6, -33.6, np.array([0.0, 0.0, 0.0]))
    # p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setAdditionalSearchPath(osp.join(os.getcwd(), 'assets'))

    # for env_num in range(100, 150):
    env_num = 22
    log_dir = '/root/data/panda_constraint/val_can_kitchen'
    env_log_dir = osp.join(log_dir, f'env_{env_num:06d}')
    # if osp.isfile(osp.join(env_log_dir, 'table_target_q.pkl')):
    #     continue
    # Reset all environments
    p_collision.resetSimulation()
    p_pcd.resetSimulation()
    p.resetSimulation()
    
    if not osp.isdir(env_log_dir):
        os.mkdir(env_log_dir)
    # Set up environment for simulation
    all_obstacles, itm_id = ipk.set_env(p, seed=env_num)
    kitchen = all_obstacles[0]
    # Set up environment for collision checking
    all_obstacles_coll, itm_id_coll = ipk.set_env(p_collision, seed=env_num)
    # Set up environment for capturing pcd
    all_obstacles_pcd, itm_id_pcd = ipk.set_env(p_pcd, seed=env_num)

    # Open the shelf
    shelf_index = 29
    p.resetJointState(all_obstacles[0], shelf_index-2, -1.57)
    p_collision.resetJointState(all_obstacles_coll[0], shelf_index-2, -1.57)
    p_pcd.resetJointState(all_obstacles_pcd[0], shelf_index-2, -1.57)

    # while True:
    #     p.stepSimulation()

    # table = p.loadURDF('table/table.urdf',[1.0,0,0], p.getQuaternionFromEuler([0,0,1.57]), useFixedBase=True)
    # all_joint_names = [
    #     pyb.getJointInfo(kitchen, i)[1].decode('utf-8')
    #     for i in range(pyb.getNumJoints(kitchen))
    # ]
    # handle_ids = [
    #     (i, pyb.getJointInfo(kitchen, i)[1].decode("utf-8")) 
    #     for i in range(pyb.getNumJoints(kitchen))
    #     if 'handle' in pyb.getJointInfo(kitchen, i)[1].decode("utf-8")
    # ]

    # Load the interactive robot
    pandaID, jointsID, fingerID = pu.set_robot(p)
    # Load the collision checking robot
    pandaID_col, jointsID_col, fingerID_col = pu.set_robot(p_collision)

    # # =================== Code for grasping objects ===========================
    # can_T_ee = np.array([[0., 0., 1, 0.], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    # # Offset by 10cm along x-axis for allowing grasping and 5 cm along z-axis for avoiding 
    # # collision w/ table.
    # can_pose = np.array(p.getBasePositionAndOrientation(itm_id)[0])+np.array([-0.11, 0.0, 0.05])
    # world_T_can = np.eye(4)
    # world_T_can[:3, 3] = can_pose
    # q, solved = get_IK_posev2(p, pandaID, jointsID, world_T_can@can_T_ee)
    # attempts = 0
    # while pu.get_distance(p, all_obstacles, pandaID)<0. or pu.check_self_collision(p, pandaID) and not solved:
    #     q_random = (pu.q_max[0]-pu.q_min[0])*np.random.rand(7)+pu.q_min[0]
    #     q, solved = get_IK_posev2(p, pandaID, jointsID, world_T_can@can_T_ee, q_random)
    #     attempts+=1
    #     if attempts>500:
    #         break
    # if solved:
    #     print("still no solution")
    #     with open(osp.join(env_log_dir, 'table_target_q.pkl'), 'wb') as f:
    #         pickle.dump({'q_goal': q}, f)

    # =================== Code for grasping with different orientations ===========================
    can_T_ee = np.array([[0., 0., 1, 0.], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    ipk.panda_reset_open_gripper(p, pandaID, gripper_dist=0.1)
    ipk.panda_reset_open_gripper(p_collision, pandaID_col, gripper_dist=0.1)
    # Offset by 10cm along x-axis for allowing grasping and 5 cm along z-axis for avoiding 
    # collision w/ table.
    can_pose = np.array(p.getBasePositionAndOrientation(itm_id)[0])
    world_T_can = smp.base.rt2tr(np.eye(3), can_pose)

    # goal_tolerance = np.array([0, 0, 0.5*np.pi])
    # # Sample random axis
    # delta_orient = (np.random.rand(3)*2-1)*goal_tolerance
    # delta_theta = np.linalg.norm(delta_orient)
    # delta_vector = delta_orient/delta_theta
    # T_offset = smp.base.rt2tr(smp.base.angvec2r(delta_theta, delta_vector), np.zeros(3))
    # world_T_offset = world_T_can@T_offset
    # # Add grasping offset for collision w/ obstacle.
    # world_T_grasp = world_T_offset@np.c_[np.eye(4, 3), np.array([-0.11, 0.0, 0.05, 1])]
    
    # q, solved = get_IK_posev2(p, pandaID, jointsID, world_T_grasp@can_T_ee)
    # attempts = 0
    # while pu.get_distance(p, all_obstacles, pandaID)<0. or pu.check_self_collision(p, pandaID) and not solved:
    #     delta_orient = (np.random.rand(3)*2-1)*goal_tolerance
    #     # Get the Axis angle form
    #     delta_theta = np.linalg.norm(delta_orient)
    #     delta_vector = delta_orient/delta_theta
    #     T_offset = smp.base.rt2tr(smp.base.angvec2r(delta_theta, delta_vector), np.zeros(3))
    #     world_T_offset = world_T_can@T_offset
    #     world_T_grasp = world_T_offset@np.c_[np.eye(4, 3), np.array([-0.11, 0.0, 0.05, 1])]
        
    #     q, solved = get_IK_posev2(p, pandaID, jointsID, world_T_grasp@can_T_ee)
    #     attempts+=1
    #     if attempts>100:
    #         break
        # if solved:
        #     print("still no solution")
        #     with open(osp.join(env_log_dir, 'table_target_q.pkl'), 'wb') as f:
        #         pickle.dump({'q_goal': q}, f)

    # # # Set up the camera:
    # view_matrix = pyb.computeViewMatrix(
    #     # cameraEyePosition=[-0.6, -0.6, 0.8],
    #     cameraEyePosition=[-0.6, -0.5, 0.7],
    #     cameraTargetPosition=[0.5, -0.4, 0.15],
    #     cameraUpVector=[0., 0., 1]
    # )
    # # # For shelf placing
    # # view_matrix = pyb.computeViewMatrix(
    # #     cameraEyePosition=[-0.6, -0.4, 1.271],
    # #     cameraTargetPosition=[0.5, -0.4, 0.721],
    # #     cameraUpVector=[0., 0., 1]
    # # )
    # fov = 45
    # height = 512
    # width = 512
    # aspect = width/height
    # near=0.02
    # far=3
    # projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # _, _, rgb_img, depth_img, seg_img = p.getCameraImage(width,
    #                             height,
    #                             view_matrix,
    #                             projection_matrix)
    # # ========== Get PCD ===========
    # pcd = get_pcd(p_pcd)
    # # o3d.visualization.draw_geometries([pcd])
    
    # ================== Trajectory planning =================
    # Define OMPL plannner
    # Planning parameters
    space = ob.RealVectorStateSpace(7)
    bounds = ob.RealVectorBounds(7)
    # Set joint limits
    for i in range(7):
        bounds.setHigh(i, pu.q_max[0, i])
        bounds.setLow(i, pu.q_min[0, i])
    space.setBounds(bounds)
    si = ob.SpaceInformation(space)
    
    # Define collison checker.
    validity_checker_obj = pu.ValidityCheckerDistance(
        p_collision,
        si, 
        all_obstacles_coll,
        pandaID_col,
        jointsID_col
    )
    # Define TSR for sampling goal poses for inputing to network
    goal_region = TSR(
        si,
        validity_checker_obj, 
        world_T_can, 
        offset_T_grasp=np.c_[np.eye(4, 3), np.array([-0.11, 0.0, 0.05, 1])]@can_T_ee, 
        goal_tolerance=np.array([0.0, 0.0, 0.5*np.pi])
    )
    # door_link_index = 29    
    # with open(f'handle_{door_link_index}_traj.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #     # Open trajectory
    #     q_traj = np.array(data['traj'])
    ipk.panda_reset_open_gripper(p, pandaID, gripper_dist=0.1)
    ipk.panda_reset_open_gripper(p_collision, pandaID_col, gripper_dist=0.1)

    # ============== Load VQ-MPT Model ======================
    dict_model_folder = '/root/data/general_mpt_panda_7d/model1'
    ar_model_folder = '/root/data/general_mpt_panda_7d/stage2/model1'
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    # Define the models
    d_model = 512
    #TODO: Get the number of keys from the saved data
    num_keys = 2048
    goal_index = num_keys + 1
    quantizer_model = VectorQuantizer(n_e=num_keys, e_dim=8, latent_dim=d_model)

    # Load quantizer model.
    dictionary_model_folder = dict_model_folder
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

    # ==================================== load model parameters =====================================

    # np.random.seed(100)
    # # Randomly sample a collision free start point.
    # initial_config = (pu.q_min + (pu.q_max-pu.q_min)*np.random.rand(7))[0]
    # pu.set_position(p_collision, pandaID_col, jointsID_col, initial_config)

    # while pu.get_distance(p_collision, all_obstacles_coll, pandaID_col)<0. or pu.check_self_collision(p_collision, pandaID_col):
    #     initial_config = (pu.q_min + (pu.q_max-pu.q_min)*np.random.rand(7))[0]
    #     pu.set_position(p_collision, pandaID_col, jointsID_col, initial_config)
    # pu.set_position(p, pandaID, jointsID, initial_config)
    # # Plan a trajectory from initial config to cupboard handle grasp location.
    # goal_q = q_traj[0]
    # traj_cupboard, _, _ , success = get_path(initial_config, goal_q, validity_checker_obj)
    # follow_trajectory(p, pandaID, jointsID, traj_cupboard)
    # j_c = pu.get_joint_position(p, pandaID, jointsID)
    # while np.linalg.norm(j_c-goal_q)<1e-12:
    #     p.setJointMotorControlArray(
    #         pandaID,
    #         jointsID,
    #         pyb.POSITION_CONTROL,
    #         targetPositions=goal_q
    #     )
    #     p.stepSimulation()
    #     j_c = pu.get_joint_position(p, pandaID, jointsID)
    
    # for _ in range(100):
    #     p.stepSimulation()
    # print(pu.get_joint_position(p, pandaID, jointsID)-goal_q)
    # for _ in range(100):
    #     panda_close_gripper(p, pandaID)
    #     p.stepSimulation()
    # for _ in range(100):
    #     p.stepSimulation()
    # follow_trajectory(p, pandaID, jointsID, q_traj)
    
    # # Open panda gripper
    # panda_open_gripper(p, pandaID, 0.1)
    # p.stepSimulation()
    
    # gripper_joint_state = p.getJointState(pandaID, 10)[0]
    # print(gripper_joint_state)
    # panda_reset_open_gripper(p_collision, pandaID_col, gripper_dist=2*gripper_joint_state)

    # # Sync collision env
    # cupboard_joint_state = p.getJointState(kitchen, door_link_index-2)[0]
    # p_collision.resetJointState(all_obstacles_coll[0], door_link_index-2, cupboard_joint_state)
    # # Sync gripper.
    # panda_reset_open_gripper(p_collision, pandaID_col, 0.1)

    # with open('shelf_reach_q.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #     can_start_q = data['start_q']
    # with open('table_target_q.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #     can_goal_q = data['q_goal']

    # tmp_start_q = pu.get_joint_position(p, pandaID, jointsID)
    # path_cupboard_2_can, _, _, success = get_path(tmp_start_q, can_start_q, validity_checker_obj)
    # # Execute cupboard trajectory
    # follow_trajectory(p, pandaID, jointsID, path_cupboard_2_can)
    # for _ in range(10):
    #     p.stepSimulation()
    # Plan a trajectory from grasp point to table

    # # Load the constraint start and goal positions
    with open('q_start_c.pkl', 'rb') as f:
        data = pickle.load(f)
        can_start_q = data['q_start']
    
    with open(osp.join(env_log_dir, 'table_target_q.pkl'), 'rb') as f:
        data = pickle.load(f)
        can_goal_q = data['q_goal']
        saved_can_goal_q = data['q_goal']
    # pu.set_position(p, pandaID, jointsID, can_goal_q)
    # =================== Code for constraint planning ===========================    
    # Find a constraint path without tilting the object
    tolerance = np.array([2*np.pi, 0.1, 0.1])
    constraint_function = pcs.EndEffectorConstraint(can_T_ee[:3, :3], tolerance, pandaID, jointsID)
    # Get point cloud information.
    pcd = ipk.get_pcd(p_pcd)
    map_data = tg_data.Data(pos=torch.as_tensor(np.asarray(pcd.points), dtype=torch.float, device=device))

    use_model = True
    test_state = ob.State(space)
    goal_samples = []
    for _ in range(10):
        goal_region.sampleGoal(test_state)
        sample_goal = np.array([test_state[i] for i in range(7)])
        goal_samples.append(sample_goal)
    goal_samples = np.r_[goal_samples]

    plan_time = 0.0
    if use_model:
        for can_goal_q in goal_samples:
            n_start_n_goal = (np.r_[can_start_q[None, :], can_goal_q[None, :]]-pu.q_min)/(pu.q_max-pu.q_min)
            search_dist_mu, search_dist_sigma, _ = ec7.get_search_dist(
                n_start_n_goal, 
                None, 
                map_data, 
                context_env_encoder, 
                decoder_model, 
                ar_model, 
                quantizer_model, 
                num_keys
            )

            # # Sampling for TSR
            # n_start = (can_start_q[None, :]-pu.q_min)/(pu.q_max-pu.q_min)
            # n_goals = (goal_samples-pu.q_min)/(pu.q_max-pu.q_min)
            # search_dist_mu, search_dist_sigma, _ = get_search_dist_batch(
            #     n_start,
            #     n_goals,
            #     map_data, 
            #     context_env_encoder, 
            #     decoder_model, 
            #     ar_model, 
            #     quantizer_model, 
            #     num_keys
            # )

            # # Visualize the output distributions
            # fig, ax_all = plt.subplots(1, 3, figsize=(12, 8.8))
            # for i, ax in enumerate(ax_all):
            #     ax.scatter(*search_dist_mu.numpy()[:, 2*i:2*(i+1)].T, label='Dist mean')
            #     ax.set_xlim(pu.q_min[0, 2*i], pu.q_max[0, 2*i])
            #     ax.set_ylim(pu.q_min[0, 2*(i+1)], pu.q_max[0, 2*(i+1)])
            # fig.show()
            # search_dist_mu, search_dist_sigma, _ = ec7.get_search_proj_dist(
            #     n_start_n_goal, 
            #     np.r_[can_start_q[None, :], can_goal_q[None, :]], 
            #     map_data, 
            #     context_env_encoder, 
            #     decoder_model, 
            #     ar_model, 
            #     quantizer_model, 
            #     num_keys
            # )
        
            path, _, plan_time, _, success = get_constraint_path_v2(
                can_start_q,
                can_goal_q, 
                validity_checker_obj, 
                constraint_function, 
                search_dist_mu, 
                search_dist_sigma,
                plan_time=5
            )
            if success:
                break
    else:
        search_dist_mu, search_dist_sigma = None, None
        path, _, plan_time, _, success = get_constraint_path(
            can_start_q, 
            world_T_can, 
            validity_checker_obj, 
            constraint_function, 
            search_dist_mu, 
            search_dist_sigma, 
            goal_samples=goal_samples, 
            # plan_time=5
        )

    # # ---------------- test - TSR's -----------------------------
    # # Planning parameters
    # space = ob.RealVectorStateSpace(7)
    # bounds = ob.RealVectorBounds(7)
    # # Set joint limits
    # for i in range(7):
    #     bounds.setHigh(i, pu.q_max[0, i])
    #     bounds.setLow(i, pu.q_min[0, i])
    # space.setBounds(bounds)

    
    # # Set up the constraint planning space.
    # css = ob.ProjectedStateSpace(space, constraint_function)
    # csi = ob.ConstrainedSpaceInformation(css)
    
    # # Define validity checker
    # ss = og.SimpleSetup(csi)
    # ss.setStateValidityChecker(validity_checker_obj)

    # # Define the start and goal state
    # test_state = ob.State(csi.getStateSpace())
    # goal_state = ob.State(csi.getStateSpace())

    # success = False
    # panda_model = rtb.models.DH.Panda()
    # ik_solver = TracIKSolver(
    #     "assets/franka_panda/franka_panda.urdf",
    #     "panda_link0",
    #     "panda_hand",
    #     # "panda_rightfinger",
    #     timeout=0.1,
    #     solve_type='Distance'
    # )
    # goal_T = ik_solver.fk(can_goal_q)
    # # goal_T = panda_model.fkine(can_goal_q).A
    # goal_region = TSR(csi, validity_checker_obj, goal_T, tolerance=np.array([0.25*np.pi, 0.0, 0.0]))
    # ompl_to_np = lambda x: np.array([x[i] for i in range(7)])
    # # ---------------- test - TSR's -----------------------------
    
    
    # path, plan_time, _, _ = get_path(can_start_q, can_goal_q, validity_checker_obj, search_dist_mu, search_dist_sigma)

    # path_can, _, _, success = get_path(can_start_q, can_goal_q, validity_checker_obj)
    # # Close the gripper
    # for _ in range(200):
    #     panda_close_gripper(p, pandaID)
    #     p.stepSimulation()
    # follow_trajectory(p, pandaID, jointsID, path_can)
    
    # # ================ Grasping code. ======================
    # vM = np.reshape(view_matrix, (4, 4), order='F')
    # pM = np.reshape(projection_matrix, (4, 4), order='F')
    # tran_pix_world = np.linalg.inv(pM@vM)
    
    # # Get point cloud data from depth.
    # pcd = depth2pc_opengl(depth_img, tran_pix_world, height, width)
    
    # # Contact-GraspNet chkpt
    # ckpt_dir = '/root/third_party/models/scene_test_2048_bs3_hor_sigma_0025'
    # forward_passes = 5
    # global_config = config_utils.load_config(ckpt_dir, batch_size=forward_passes)
    # # Build the model
    # grasp_estimator = GraspEstimator(global_config)
    # grasp_estimator.build_network()
    # # Add ops to save and restore all the variables.
    # saver = tf.train.Saver(save_relative_paths=True)
    # # Create a session
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    # sess = tf.Session(config=config)
    # # Load weights
    # grasp_estimator.load_weights(sess, saver, ckpt_dir, mode="test")
    # # NOTE: Produced grasps are in camera frame!!

    # # Translate pcd (world frame) to camera frame
    # view_pcd = (np.c_[pcd, np.ones(pcd.shape[0])]@vM.T)[:, :3]
    # # Rotate OpenGL co-ordinates to CV format
    # R_VM_2_CV = np.array([
    #     [1, 0, 0],
    #     [0, -1.0, 0],
    #     [0, 0, -1.0]
    # ])
    # cv_pcd = view_pcd@R_VM_2_CV.T
    # pc_segmap = seg_img.ravel(order='C')
    # pc_segments = {}
    # for i in np.unique(pc_segmap):
    #     if i == 0:
    #         continue
    #     else:
    #         pc_segments[i] = cv_pcd[pc_segmap == i]
    # # Predict grasping points
    # pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(
    #     sess,
    #     cv_pcd,
    #     pc_segments=pc_segments,
    #     local_regions=True,
    #     filter_grasps=True,
    #     forward_passes=forward_passes,
    # )
    # pc_colors = rgb_img[:, :, :3].reshape(-1, 3)
    # # Rotate along z-axis the grasp orientation (in the grasp frame)
    # # by 180deg.
    # inverted_pred_grasps = {}
    # inv_T = np.eye(4)
    # inv_T[0, 0] = -1.0
    # inv_T[1, 1] = -1.0
    # for key in pred_grasps_cam.keys():
    #     inverted_pred_grasps[key] = pred_grasps_cam[key]@inv_T[None, :, :]
    
    # # Visualize PC and grasping data.
    # # visualize_grasps(cv_pcd, pred_grasps_cam, scores, pc_colors=pc_colors, plot_opencv_cam=True)
    # visualize_grasps(cv_pcd, inverted_pred_grasps, scores, pc_colors=pc_colors, plot_opencv_cam=True)
    
    # # ========== Find a grasp for which an IK exists ========================
    # panda_reset_open_gripper(pandaID, gripper_dist=0.1)
    # grasp_index_asc = np.argsort(scores[3])[::-1]
    # for grasp_index in grasp_index_asc:
    #     # OpenCV to grasp 

    #     T_CV_G = pred_grasps_cam[3][grasp_index]
    #     # Camera co-ordinate to grasp
    #     T_C_G = np.r_[R_VM_2_CV.T@T_CV_G[:3, :], np.array([[0, 0, 0, 1.0]])]
    #     # In future use better ways to invert transformation matrix.
    #     T_W_G = np.linalg.inv(vM)@T_C_G

    #     grasp_orient = Rot.from_matrix(T_W_G[:3, :3]).as_quat()
    #     # Find IK for the given tranformation:
    #     # q, solved = get_IK_pose(pandaID, jointsID, ee_pose=T_W_G[:3, 3], ee_orient=grasp_orient, link_id=8)
    #     q, solved = get_IK_posev2(pandaID, jointsID, T_W_G)
    #     if solved:
    #         break
    #     # Check if grasp exists for inverted orientation
    #     print("Checking for inverted grasp")
    #     # OpenCV to grasp 
    #     T_CV_G = inverted_pred_grasps[3][grasp_index]
    #     # Camera co-ordinate to grasp
    #     T_C_G = np.r_[R_VM_2_CV.T@T_CV_G[:3, :], np.array([[0, 0, 0, 1.0]])]
    #     # In future use better ways to invert transformation matrix.
    #     T_W_G = np.linalg.inv(vM)@T_C_G
        
    #     grasp_orient = Rot.from_matrix(T_W_G[:3, :3]).as_quat()
    #     # Find IK for the given tranformation:
    #     # q, solved = get_IK_pose(pandaID, jointsID, ee_pose=T_W_G[:3, 3], ee_orient=grasp_orient, link_id=8)
    #     q, solved = get_IK_posev2(pandaID, jointsID, T_W_G)
    #     if solved:
    #         break
    # ========================================================================
    # Find the "BEST" grasp with the highest scores for objects on the shelf:
    # grasp_index = np.argmax(scores[3]) # Index 3 is the chips can
    # T_CV_G = pred_grasps_cam[3][grasp_index]
    # T_C_G = np.r_[R_VM_2_CV.T@T_CV_G[:3, :], np.array([[0, 0, 0, 1.0]])]
    # # In future use better ways to invert transformation matrix.
    # T_W_G = np.linalg.inv(vM)@T_C_G
    
    # # Find IK for the given tranformation:
    # panda_reset_open_gripper(pandaID, gripper_dist=0.1)
    # grasp_orient = Rot.from_matrix(T_W_G[:3, :3]).as_quat()
        # solved = False
    # while not solved:
    #     q, solved = get_IK_pose(pandaID, jointsID, ee_pose=T_W_G[:3, 3], ee_orient=grasp_orient, link_id=8)
    # ============================== best grasp ==============================

    # ---------------------------- end-of-grasping --------------------------------
    
    # # Save data for testing.
    # with open('depth_img.pkl', 'wb') as f:
    #     pickle.dump(
    #     {
    #         'rbg_img': rgb_img,
    #         'depth_img': depth_img,
    #         'seg_img': seg_img,
    #         'near': near,
    #         'far': far
    #     }, f)
    # # Identify grasping locations:
    # while True:
    #     p.stepSimulation()
    #============================ Trajectories for opening/closing the shelf ===================
    # # Identify grasping locations within the scene
    # # NOTE: Left/Right are viwed after the kitchen is rotated by 180 around z-axis.
    # # Hence there may be a discrepancy w/ the name of the joint and definitions here
    # # Left draw handles - 39(top), 42(bottom)
    # # Left top cupboards - 20(right), 24(middle), 16 (left)
    # # Right draw handles - 57(top), 59(bottom)
    # # Right top cupboards - 33(right), 29 (left)

    # construct a trajectory for the draw
    link_index = 29
    
    # shelf_traj = []
    # shelf_orient = []
    # # panda_reset_close_gripper(pandaID)
    # # # For drawer
    # # for i in np.linspace(0, 0.35, 20):
    # #     link_joint_index = link_index-1
    # # # For cupboard
    # # for i in np.linspace(-1.57, 0, 40)[::-1]:
    # shelf_joint_traj = np.linspace(-np.pi*0.45, 0, 80)[::-1]
    # for i in shelf_joint_traj:
    #     link_joint_index = link_index-2
    #     p.resetJointState(kitchen, link_joint_index, i)
    #     shelf_traj.append(np.array(p.getLinkState(kitchen, link_index)[0]))
    #     shelf_orient.append(np.array(p.getLinkState(kitchen, link_index)[1]))
    # shelf_traj = np.array(shelf_traj)
    # shelf_orient = np.array(shelf_orient)
    # p.resetJointState(kitchen, link_joint_index, 0.0)
    # # # NOTE: Running sim helps with grasping stability!!
    # for _ in range(10):
    #     p.stepSimulation()
    # # choose a handle pose that the robot can reach.
    # handle_pose = np.array(p.getLinkState(kitchen, link_index)[0])
    # # # add_debug_point(p, handle_pose[None, :])
    # handle_orient = [0.0, np.pi/2, 0.0]
    # handle_orient_q = pyb.getQuaternionFromEuler(handle_orient)
    # # # For draw
    # # R_H_T = np.array(
    # #     [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    # # )
    # # For cabinet
    # R_H_T = np.array([
    #     [0, 0, -1], [0, -1, 0], [-1, 0, 0]
    # ])
    # inv_R = np.eye(3)
    # inv_R[0, 0] = -1.0
    # inv_R[1, 1] = -1.0
    # R_handle = Rot.from_quat(pyb.getLinkState(kitchen, link_index)[1])
    # # R_target = R_handle.as_matrix()@R_H_T  # Normal grasp
    # R_target = R_handle.as_matrix()@inv_R@R_H_T # Inverted grasp
    # handle_orient_q = Rot.from_matrix(R_target).as_quat()
    # handle_pose = handle_pose + np.array([-0.2, 0.0, 0.])
    # panda_reset_open_gripper(p, pandaID, 0.025)
    # # find the joint angles that will open the draw
    # q_traj = []
    # q_i = None
    # for i, handle_pose in enumerate(shelf_traj):
    #     R_handle = Rot.from_quat(shelf_orient[i])
    #     R_target = R_handle.as_matrix()@R_H_T
    #     T_target = np.eye(4)
    #     T_target[:3, :3] = R_target
    #     # Offset grasping point by -0.2 if using TracIK Solver.
    #     T_target[:3, 3] = handle_pose + R_target@np.array([0., 0.0, -0.1]).T
    #     handle_orient_q = Rot.from_matrix(R_target).as_quat()
    #     q, _ = get_IK_posev2(p, pandaID, jointsID, T_target, q_i)
    #     # set q_i for the first joint value.
    #     if q_i is None:
    #         q_i = q
    #     # q, _ = get_IK_pose(p, pandaID, jointsID, handle_pose, handle_orient_q, q_i)
    #     # Check if robot is in collision w/ itself or kitchen and close to the previous solution.
    #     not_reached=False
    #     not_reached = True
    #     # print(f"Joint distance : {np.linalg.norm(q_i-q)}"+ f" Cartesian distance : {np.linalg.norm(np.array(p.getLinkState(pandaID, 11)[0])-handle_pose)}")
    #     if np.linalg.norm(np.array(p.getLinkState(pandaID, 11)[0])-handle_pose)<1e-2:
    #         not_reached = False
    #     while pu.get_distance(p, all_obstacles, pandaID)<0. or pu.check_self_collision(p, pandaID) or not_reached:
    #         # q, _ = get_IK_pose(p, pandaID, jointsID, handle_pose, handle_orient_q, q_i)
    #         # q, _ = get_IK_posev2(p, pandaID, jointsID, T_target, q_i)
    #         q, _ = get_IK_posev2(p, pandaID, jointsID, T_target, None)
    #         not_reached = True
    #         # print(f"Joint distance : {np.linalg.norm(q_i-q)}"+ f" Cartesian distance : {np.linalg.norm(np.array(p.getLinkState(pandaID, 11)[0])-handle_pose)}")
    #         if np.linalg.norm(np.array(p.getLinkState(pandaID, 11)[0])-handle_pose)<1e-2:
    #             not_reached = False
    #     print(np.linalg.norm(np.array(p.getLinkState(pandaID, 11)[0])-handle_pose))
    #     q_i = q
    #     q_traj.append(np.array(q)[:7])
    # pu.set_position(p, pandaID, jointsID, q_traj[0])
    # for q_i in q_traj:
    #     pu.set_position(p, pandaID, jointsID, q_i)
    #     time.sleep(0.1)
    # with open(f'handle_{link_index}_traj.pkl', 'wb') as f:
    #     pickle.dump({'traj':q_traj}, f)
    # pyb.resetJointState(kitchen, link_joint_index, 0.35)
    # with open(f'handle_{link_index}_traj.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # # Open trajectory
    # q_traj = np.array(data['traj'])
    
    # # panda_reset_open_gripper(p, pandaID)
    # # Place the arm at the handle
    # pu.set_position(p, pandaID, jointsID, q_traj[0])
    # for _ in range(100):
    #     panda_close_gripper(p, pandaID)
    #     p.stepSimulation()
    # follow_trajectory(p, pandaID, jointsID, q_traj)
    
    # Identify re-grasping points.
    # dis_index = np.where(np.linalg.norm(np.diff(q_traj, axis=0), axis=1)>1)[0]

    # # connect_traj, _, _, success = get_path(q_traj[dis_index], q_traj[dis_index+1], validity_checker_obj)
    # # connect_traj = np.array()
    # # # Close trajectory
    # # q_traj = data['traj'][::-1]
    # # # execute the trajectory
    # # # ------------- Move the robot ------------------------------------
    # panda_reset_open_gripper(p, pandaID)
    # # Place the arm at the handle
    # pu.set_position(p, pandaID, jointsID, q_traj[0])
    # for start_index, stop_index in zip(np.r_[0, dis_index+1], np.r_[dis_index, q_traj.shape[0]]):
    #     for _ in range(100):
    #         panda_close_gripper(p, pandaID)
    #         p.stepSimulation()
    #     # for q_i in q_traj[start_index:stop_index]:
    #     #     pu.set_position(p, pandaID, jointsID, q_i)
    #     #     time.sleep(0.1)
    #     follow_trajectory(p, pandaID, jointsID, q_traj[start_index:stop_index])
    #     # Open end-effector links
    #     for _ in range(10):
    #         panda_open_gripper(p, pandaID, 0.1)
    #         p.stepSimulation()
    #     # Open end-effector links
    #     # panda_reset_open_gripper(p, pandaID, 0.025)
    #     panda_reset_open_gripper(p_collision, pandaID_col, 0.1)
    #     # Sync cupboard state.
    #     cupboard_joint_state = p.getJointState(kitchen, link_index-2)[0]
    #     p_collision.resetJointState(all_obstacles_coll[0], link_index-2, cupboard_joint_state)
    #     if stop_index<q_traj.shape[0]:
    #         connect_traj, _, _, success = get_path(q_traj[stop_index], q_traj[stop_index+1], validity_checker_obj)
    #         follow_trajectory(p, pandaID, jointsID, connect_traj)
