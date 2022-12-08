''' A script to test the planning model.
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

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    raise "Run code from a container with OMPL installed"

from modules.quantizers import VQEmbeddingEMA
from modules.decoder import DecoderPreNorm
from modules.encoder import EncoderPreNorm

from modules.autoregressive import AutoRegressiveModel, EnvContextCrossAttModel

from utils import ValidityChecker

res = 0.05
device = 'cpu' if torch.cuda.is_available() else torch.device('cuda')

eps = 1e-4
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
        self.qMin = qMin
        self.qMax = qMax
        self.seq_num = dist_mu.shape[0]
        if dist_mu is None:
            self.X = None
            self.U = stats.uniform(np.zeros_like(qMin), np.ones_like(qMax))
        else:
            self.X = MultivariateNormal(dist_mu,torch.diag_embed(dist_sigma+eps))
                       
    def get_random_samples(self):
        '''Generates a random sample from the list of points
        '''
        index = 0
        random_samples = np.random.permutation(self.X.sample()*24)
        while True:
            yield random_samples[index, :]
            index += 1
            if index==self.seq_num:
                random_samples = np.random.permutation(self.X.sample()*24)
                index = 0
                
    def sampleUniform(self, state):
        '''Generate a sample from uniform distribution or key-points
        :param state: ompl.base.Space object
        '''
        if self.X is None:
            sample_pos = (self.qMax-self.qMin)*self.U.rvs()+self.qMin
        else:
            sample_pos = next(self.get_random_samples())
        for i, val in enumerate(sample_pos):
            state[i] = np.float(val)
        return True


def get_ompl_state(space, state):
    ''' Returns an OMPL state
    '''
    ompl_state = ob.State(space)
    ompl_state[0] = state[0]
    ompl_state[1] = state[1]
    return ompl_state


def get_path(start, goal, input_map, dist_mu, dist_sigma):
    '''
    Plan a path given the start, goal and patch_map.
    :param start:
    :param goal:
    :param patch_map:
    :param plannerType: The planner type to use
    :param cost: The cost of the path
    :param exp: If exploration is enabled
    returns bool: Returns True if a path was planned successfully.
    '''
    mapSize = input_map.shape
    # Planning parametersf
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0.0)
    bounds.setHigh(0, mapSize[1]*res) # Set width bounds (x)
    bounds.setHigh(1, mapSize[0]*res) # Set height bounds (y)
    space.setBounds(bounds)

    state_sampler = partial(StateSamplerRegion, dist_mu=dist_mu, dist_sigma=dist_sigma, qMin=np.array([0, 0]), qMax=np.array([24, 24]))
    space.setStateSamplerAllocator(ob.StateSamplerAllocator(state_sampler))

    si = ob.SpaceInformation(space)
    validity_checker_obj = ValidityChecker(si, input_map)
    si.setStateValidityChecker(validity_checker_obj)

    start_state = get_ompl_state(space, start)
    goal_state = get_ompl_state(space, goal)

    success = False

    # Define planning problem
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_state, goal_state)

    planner = og.RRT(si)

    # if plannerType=='rrtstar':
    #     planner = og.RRTstar(si)
    # elif plannerType=='informedrrtstar':
    #     planner = og.InformedRRTstar(si)
    # elif plannerType=='bitstar':
    #     planner = og.BITstar(si)
    #     planner.setSamplesPerBatch(100)
    # else:
    #     raise TypeError(f"Planner Type {plannerType} not found")
    
    # Set the problem instance the planner has to solve

    planner.setProblemDefinition(pdef)
    planner.setup()

    # Attempt to solve the planning problem in the given time
    start_time = time.time()
    solved = planner.solve(1.0)
    plan_time = time.time()-start_time
    plannerData = ob.PlannerData(si)
    planner.getPlannerData(plannerData)
    numVertices = plannerData.numVertices()

    if pdef.hasExactSolution():
        success = True

        print("Found Solution")
        path = [
            [pdef.getSolutionPath().getState(i)[0], pdef.getSolutionPath().getState(i)[1]]
            for i in range(pdef.getSolutionPath().getStateCount())
            ]
    else:
        path = [[start[0], start[1]], [goal[0], goal[1]]]

    return path, plan_time, numVertices, success


def get_beam_search_path(max_length, K, context_output, ar_model, quantizer_model):
    ''' A beam search function, that stops when any of the paths hits termination.
    :param max_length: Max length to search.
    :param K: Number of paths to keep.
    :param context_output: the tensor ecoding environment information.
    :param ar_model: nn.Model type for the Auto-Regressor.
    :param quantizer_model: For extracting the feature vector.
    '''
    
    # Create place holder for input sequences.`
    input_seq = torch.ones(K, max_length, 512, dtype=torch.float, device=device)*-1
    quant_keys = torch.ones(K, max_length)*-1
    mask = torch.zeros(K, max_length+2)
            
        
    ar_model_input_i = torch.cat([context_output.repeat((K ,1, 1)), input_seq], dim=1)
    # mask the start/goal encoding and the prev. sequences.
    mask[:, :3] = 1

    # Get first set of quant_keys
    ar_output = ar_model(ar_model_input_i, mask)
    path_cost, start_index = F.log_softmax(ar_output[:, 2, :], dim=-1).topk(k=K, dim=-1)
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
        new_sequence = torch.as_tensor(np.array(np.unravel_index(flatten_index.numpy(), seq_cost.shape)).T)

        # Update previous keys given the current prediction.
        quant_keys[:, :i] = quant_keys[new_sequence[:, 0], :i]
        # Update the current set of keys.
        quant_keys[:, i] = new_sequence[:, 1].to(dtype=torch.float)
        # Update the cost
        path_cost = nxt_cost

        # Break at the first sign of termination
        if (new_sequence[:, 1] == 1025).any():
            break

        # Select index
        select_index = new_sequence[:, 1] !=1025

        # Update the input embedding. 
        input_seq[select_index, :i+1, :] = input_seq[new_sequence[select_index, 0], :i+1, :]
        input_seq[select_index, i+1, :] = quantizer_model.output_linear_map(quantizer_model.embedding(new_sequence[select_index, 1]))
    return quant_keys, path_cost, input_seq

def main(args):
    ''' Main running script.
    :parma args: An object of type argparse.ArgumentParser().parse_args()
    '''
    # ========================= Load trained model ===========================
    # Define the models
    d_model = 512
    quantizer_model = VQEmbeddingEMA(n_e=1024, e_dim=8, latent_dim=d_model)

    # Load quantizer model.
    dictionary_model_folder = args.dict_model_folder
    with open(osp.join(dictionary_model_folder, 'model_params.json'), 'r') as f:
        dictionary_model_params = json.load(f)

    encoder_model = EncoderPreNorm(**dictionary_model_params)
    decoder_model = DecoderPreNorm(
        e_dim=dictionary_model_params['d_model'], 
        h_dim=dictionary_model_params['d_inner'], 
        c_space_dim=dictionary_model_params['c_space_dim']
    )

    checkpoint = torch.load(osp.join(dictionary_model_folder, 'best_model.pkl'))
    
    # Load model parameters and set it to eval
    for model, state_dict in zip([encoder_model, quantizer_model, decoder_model], ['encoder_state', 'quantizer_state', 'decoder_state']):
        model.load_state_dict(checkpoint[state_dict])
        model.eval()

    # Load the AR model.
    # NOTE: Save these values as dictionary in the future, and load as json.
    env_params = {
        'd_model': dictionary_model_params['d_model'],
        'dropout': 0.1,
        'n_position': 40*40
    }
    ar_model_folder = args.ar_model_folder
    # Create the environment encoder object.
    with open(osp.join(ar_model_folder, 'cross_attn.json'), 'r') as f:
        context_env_encoder_params = json.load(f)
    context_env_encoder = EnvContextCrossAttModel(env_params, context_env_encoder_params)
    # Create teh AR model
    with open(osp.join(ar_model_folder, 'ar_params.json'), 'r') as f:
        ar_params = json.load(f)
    ar_model = AutoRegressiveModel(**ar_params)

    # Load the parameters and set the model to eval
    checkpoint = torch.load(osp.join(ar_model_folder, 'best_model.pkl'))
    for model, state_dict in zip([context_env_encoder, ar_model], ['context_state', 'ar_model_state']):
        model.load_state_dict(checkpoint[state_dict])
        model.eval()

    # ============================= Run planning experiment ============================
    val_data_folder = args.val_data_folder
    start = args.start
    pathSuccess = []
    pathTime = []
    pathTimeOverhead = []
    pathVertices = []
    predict_seq_time = []
    for env_num in range(start, start+args.samples):
        print(env_num)
        map_file = osp.join(val_data_folder, f'env{env_num:06d}/map_{env_num}.png')
        env_map = skimage.io.imread(map_file, as_gray=True)
        for path_num in range(args.num_paths):
            path_file = osp.join(val_data_folder, f'env{env_num:06d}/path_{path_num}.p')
            data = pickle.load(open(path_file, 'rb'))
            path = data['path']
            if data['success']:
                # Get the context.
                start_time = time.time()
                start_n_goal = torch.as_tensor(path[[0, -1], :]/25, dtype=torch.float)
                env_input = torch.as_tensor(env_map[None, :], dtype=torch.float)
                context_output = context_env_encoder(env_input[None, :].to(device), start_n_goal[None, :].to(device))
                # Find the sequence of dict values using beam search
                quant_keys, _, input_seq = get_beam_search_path(51, 3, context_output, ar_model, quantizer_model)
                reached_goal = torch.stack(torch.where(quant_keys==1025), dim=1)
                s = False
                # if reached_goal.shape[1] > 0:
                if len(reached_goal) > 0:
                    # Get the distribution.
                    output_dist_mu, output_dist_sigma = decoder_model(input_seq[reached_goal[0, 0], :reached_goal[0, 1]])
                    dist_mu = output_dist_mu.detach().cpu().squeeze()
                    dist_sigma = output_dist_sigma.detach().cpu().squeeze()
                    if len(dist_mu.shape) == 1:
                        dist_mu = dist_mu[None, :]
                        dist_sigma = dist_sigma[None, :]
                    patch_time = time.time() - start_time
                    _, t, v, s = get_path(path[0], path[-1], env_map, dist_mu, dist_sigma)
                    pathSuccess.append(s)
                    pathTime.append(t)
                    pathVertices.append(v)
                    pathTimeOverhead.append(t)
                    predict_seq_time.append(patch_time)
                else:
                    pathSuccess.append(False)
                    pathTime.append(0)
                    pathVertices.append(0)
                    pathTimeOverhead.append(0)
                    predict_seq_time.append(0)
    
    pathData = {'Time':pathTime, 'Success':pathSuccess, 'Vertices':pathVertices, 'PlanTime':pathTimeOverhead, 'PredictTime': predict_seq_time}
    fileName = osp.join(ar_model_folder, f'eval_val_plan_{args.map_type}_{start:06d}.p')
    pickle.dump(pathData, open(fileName, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_model_folder', help="Folder where dictionary model is stored")
    parser.add_argument('--ar_model_folder', help="Folder where AR model is stored")
    parser.add_argument('--val_data_folder', help="Folder where environment data is stored")
    parser.add_argument('--start', help="Env number to start", type=int)
    parser.add_argument('--samples', help="Number of samples to collect", type=int)
    parser.add_argument('--num_paths', help="Number of paths for each environment", type=int)
    parser.add_argument('--map_type', help="Type of map", choices=['forest', 'maze'])

    args = parser.parse_args()
    main(args)