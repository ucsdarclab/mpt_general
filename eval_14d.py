''' Code for evaluating 14D robot
'''

''' A script to test the planning model for panda
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
import json
import argparse
import pickle
import open3d as o3d
import torch_geometric.data as tg_data

import matplotlib.pyplot as plt

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    raise "Run code from a container with OMPL installed"

from os import path as osp
import sys
sys.path.insert(0, osp.abspath(osp.join(osp.curdir, 'dual_arms')))

from collect_data import set_env
import dual_arm_utils as dau
from modules.quantizers import VectorQuantizer
from modules.decoder import DecoderPreNorm, DecoderPreNormGeneral
from modules.encoder import EncoderPreNorm

from modules.autoregressive import AutoRegressiveModel, EnvContextCrossAttModel

from panda_utils import get_pybullet_server, q_max, q_min
from panda_shelf_env import place_shelf

def get_ompl_state(space, state):
    ''' Returns an OMPL state
    '''
    ompl_state = ob.State(space)
    for i in range(14):
        ompl_state[i] = state[i]
    return ompl_state

def get_numpy_state(state):
    ''' Return the state as a numpy array.
    :param state: An ob.State from ob.RealVectorStateSpace
    :return np.array:
    '''
    return np.array([state[i] for i in range(14)])

res = 0.05
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

q_bi_max = np.c_[q_max, q_max]
q_bi_min = np.c_[q_min, q_min]

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
            state[i] = np.float(val)
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

def get_path(start, goal, env_num, dist_mu=None, dist_sigma=None, cost=None, planner_type='rrtstar'):
    '''
    Plan a path given the start, goal and patch_map.
    :param start:
    :param goal:
    :param env_num:
    :param dist_mu:
    :param dist_sigma:
    :param cost:
    :param planner_type:
    returns (list, float, int, bool): Returns True if a path was planned successfully.
    '''
    # Planning parameters
    space = ob.RealVectorStateSpace(14)
    bounds = ob.RealVectorBounds(14)
    # Set joint limits
    for i in range(14):
        bounds.setHigh(i, q_bi_max[0, i])
        bounds.setLow(i, q_bi_min[0, i])
    space.setBounds(bounds)

    # # Redo the state sampler
    state_sampler = partial(StateSamplerRegion, dist_mu=dist_mu, dist_sigma=dist_sigma, qMin=q_bi_min, qMax=q_bi_max)
    space.setStateSamplerAllocator(ob.StateSamplerAllocator(state_sampler))

    si = ob.SpaceInformation(space)
    p = get_pybullet_server('direct')
    # Env - random objects

    si = ob.SpaceInformation(space)
    robotid1, robotid2, all_obstacles = set_env(p, env_num)
    validity_checker_obj = dau.ValidityCheckerDualDistance(
        si,
        robotID_1=(robotid1[0], robotid1[1]),
        robotID_2=(robotid2[0], robotid2[1]),
        obstacles=all_obstacles
    )
    si.setStateValidityChecker(validity_checker_obj)

    start_state = get_ompl_state(space, start)
    goal_state = get_ompl_state(space, goal)

    success = False

    # Define planning problem
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_state, goal_state)

    # Set up objective function
    obj = getPathLengthObjective(cost, si)
    pdef.setOptimizationObjective(obj)

    if planner_type=='rrtstar':
        planner = og.RRTstar(si)
        planner.setRange(13)
    elif planner_type=='informedrrtstar':
        planner = og.InformedRRTstar(si)
    elif planner_type=='bitstar':
        planner = og.BITstar(si)
        planner.setSamplesPerBatch(100)
    elif planner_type=='fmtstar':
        planner = og.FMT(si)
    elif planner_type=='rrtconnect':
        planner = og.RRTConnect(si)
        planner.setRange(13)
    else:
        print("None of the planners found, using RRT")
        planner = og.RRT(si)
        planner_type = 'rrt'
        planner.setRange(13.07)
    #     raise TypeError(f"Planner Type {plannerType} not found")

    # planner.setRange(0.1)
    # Set the problem instance the planner has to solve

    planner.setProblemDefinition(pdef)
    planner.setup()

    # Attempt to solve the planning problem in the given time
    start_time = time.time()
    solved = planner.solve(5.0)
    current_time = 5.0
    while (not pdef.hasOptimizedSolution() and current_time<250) and not pdef.hasExactSolution():
        # Only solve for path if there is a solution
        if pdef.hasExactSolution():
            # do path simplification
            path_simplifier = og.PathSimplifier(si)
            # using try catch here, sometimes path simplification produces
            # core dumped errors.
            try:
                path_simplifier.simplify(pdef.getSolutionPath(), 0.0)
                print(f"After path simplification, path length: {pdef.getSolutionPath().length()}")
                if pdef.getSolutionPath().length()<=cost:
                    break
            except TypeError:
                print("Path not able to simplify because no solution found!")
                pass
        solved = planner.solve(1)
        current_time = time.time()-start_time
    # if not pdef.hasExactSolution():
    #     # Redo the state sampler
    #     state_sampler = partial(StateSamplerRegion, dist_mu=None, dist_sigma=None, qMin=q_min, qMax=q_max)
    #     space.setStateSamplerAllocator(ob.StateSamplerAllocator(state_sampler))
    #     solved = planner.solve(25.0)
    plan_time = time.time()-start_time
    plannerData = ob.PlannerData(si)
    planner.getPlannerData(plannerData)
    numVertices = plannerData.numVertices()

    if pdef.hasExactSolution():
        success = True
        # Simplify solution
        path_simplifier = og.PathSimplifier(si)
        try:
            path_simplifier.simplify(pdef.getSolutionPath(), 0.0)
        except TypeError:
            print("Path not able to simplify for unknown reason!")
            pass
        print("Found Solution")
        # TODO: Get final planner path. 
        path = [
            get_numpy_state(pdef.getSolutionPath().getState(i))
            for i in range(pdef.getSolutionPath().getStateCount())
            ]
    else:
        path = [start, goal]

    # num_vertices = plannerData.numVertices()
    # sampled_vertex = np.array([get_numpy_state(plannerData.getVertex(i).getState()) for i in range(num_vertices)])
    # approx_path =np.array([
    #     get_numpy_state(pdef.getSolutionPath().getState(i))
    #     for i in range(pdef.getSolutionPath().getStateCount())
    # ])
    # fig, ax_all = plt.subplots(1,3, figsize=(24, 8))
    # for i, ax in enumerate(ax_all):
    #     ax.scatter(*sampled_vertex[:, 2*i:2*(i+1)].T, color='g', alpha=0.4)
    #     ax.plot(*approx_path[:, 2*i:2*(i+1)].T, linewidth=2)
    #     ax.set_xlim([q_min[0, 2*i], q_max[0, 2*i]])
    #     ax.set_ylim([q_min[0, 2*i +1], q_max[0, 2*i+1]])
    # plt.savefig(f'sampled_vertex_{env_num}.png')
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
    start_n_goal = torch.as_tensor(normalized_path[[0, -1]], dtype=torch.float)
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
        search_dist_mu = torch.zeros((reached_goal[0, 1]+1, 14))
        search_dist_mu[:reached_goal[0, 1], :] = dist_mu
        search_dist_mu[reached_goal[0, 1], :] = torch.tensor(normalized_path[-1])
        search_dist_sigma = torch.diag_embed(torch.ones((reached_goal[0, 1]+1, 14)))
        search_dist_sigma[:reached_goal[0, 1], :, :] = torch.tensor(dist_sigma)
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

        map_file = osp.join(val_data_folder, f'env_{env_num:06d}/map_{env_num}.ply')
        data_PC = o3d.io.read_point_cloud(map_file, format='ply')
        depth_points = np.array(data_PC.points)
        map_data = tg_data.Data(pos=torch.as_tensor(depth_points, dtype=torch.float, device=device))

        for path_num in range(args.num_paths):
            path_file = osp.join(val_data_folder, f'env_{env_num:06d}/path_{path_num}.p')
            data = pickle.load(open(path_file, 'rb'))
            path = (data['path']-q_bi_min)/(q_bi_max-q_bi_min)
            path_obj = np.linalg.norm(np.diff(data['path'], axis=0), axis=1).sum()
            if data['success']:
                if use_model:
                    search_dist_mu, search_dist_sigma, patch_time = get_search_dist(path, data['path'], map_data, context_env_encoder, decoder_model, ar_model, quantizer_model, num_keys)
                else:
                    search_dist_mu, search_dist_sigma, patch_time = None, None, 0.0
            
                planned_path, t, v, s = get_path(data['path'][0], data['path'][-1], env_num, search_dist_mu, search_dist_sigma, cost=path_obj, planner_type=args.planner_type)
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
        fileName = osp.join(ar_model_folder, f'eval_val_plan_{args.planner_type}_{start:06d}.p')
    else:
        fileName = f'/root/data/general_mpt_bi_panda/{args.planner_type}_{start:06d}.p'
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

    args = parser.parse_args()
    main(args)