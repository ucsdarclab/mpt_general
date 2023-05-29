#!/usr/bin python3.8
''' A script to plan using 7D robot
'''
import rospy
from moveit_msgs.srv import GetSamplingDistributionSequence, GetSamplingDistributionSequenceResponse
from moveit_msgs.msg import SamplingDistribution
from sensor_msgs.msg import PointCloud2
from utils import pointcloud2_to_xyz_array

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

from modules.quantizers import VectorQuantizer
from modules.decoder import DecoderPreNorm, DecoderPreNormGeneral
from modules.encoder import EncoderPreNorm

from modules.autoregressive import AutoRegressiveModel, EnvContextCrossAttModel

import fetch_utils as fu

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


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

def get_search_dist(normalized_path, map_data, context_encoder, decoder_model, ar_model, quantizer_model, num_keys):
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

    reached_goal = torch.stack(torch.where(quant_keys==goal_index), dim=1)
    if len(reached_goal) > 0:
        # Get the distribution.
        # Ignore the zero index, since it is encoding representation of start vector.
        output_dist_mu, output_dist_sigma = decoder_model(input_seq[reached_goal[0, 0], 1:reached_goal[0, 1]+1][None, :])
        dist_mu = output_dist_mu.detach().cpu()
        dist_sigma = output_dist_sigma.detach().cpu()
        # If only a single point is predicted, then reshape the vector to a 2D tensor.
        if len(dist_mu.shape) == 1:
            dist_mu = dist_mu[None, :]
            dist_sigma = dist_sigma[None, :]
        # ========================== append search with goal  ======================
        search_dist_mu = torch.zeros((reached_goal[0, 1]+1, 7))
        search_dist_mu[:reached_goal[0, 1], :] = dist_mu
        search_dist_mu[reached_goal[0, 1], :] = torch.tensor(normalized_path[-1])
        search_dist_sigma = torch.diag_embed(torch.ones((reached_goal[0, 1]+1, 7)))
        search_dist_sigma[:reached_goal[0, 1], :, :] = torch.tensor(dist_sigma)
        search_dist_sigma[reached_goal[0, 1], :, :] = search_dist_sigma[reached_goal[0, 1], :, :]*0.01
        # ==========================================================================
    else:
        search_dist_mu = None
        search_dist_sigma = None
    patch_time = time.time()-start_time
    return search_dist_mu, search_dist_sigma, patch_time

class DistributionSequencePredServer:
    def __init__(self):
        # ======================== Model loading :START ========================================
        # Define the models
        d_model = 512
        #TODO: Get the number of keys from the saved data
        self.num_keys = 2048
        goal_index = self.num_keys + 1
        self.quantizer_model = VectorQuantizer(n_e=self.num_keys, e_dim=8, latent_dim=d_model)

        # Load quantizer model.
        dictionary_model_folder = 'fetch_models/stage1'
        with open(osp.join(dictionary_model_folder, 'model_params.json'), 'r') as f:
            dictionary_model_params = json.load(f)

        encoder_model = EncoderPreNorm(**dictionary_model_params)
        self.decoder_model = DecoderPreNormGeneral(
            e_dim=dictionary_model_params['d_model'], 
            h_dim=dictionary_model_params['d_inner'], 
            c_space_dim=dictionary_model_params['c_space_dim']
        )

        checkpoint = torch.load(osp.join(dictionary_model_folder, 'best_model.pkl'))
        
        # Load model parameters and set it to eval
        for model, state_dict in zip([encoder_model, self.quantizer_model, self.decoder_model], ['encoder_state', 'quantizer_state', 'decoder_state']):
            model.load_state_dict(checkpoint[state_dict])
            model.eval()
            model.to(device)

        # Load the AR model.
        # NOTE: Save these values as dictionary in the future, and load as json.
        env_params = {
            'd_model': dictionary_model_params['d_model'],
        }
        ar_model_folder = 'fetch_models/stage2'
        # Create the environment encoder object.
        with open(osp.join(ar_model_folder, 'cross_attn.json'), 'r') as f:
            context_env_encoder_params = json.load(f)
        self.context_env_encoder = EnvContextCrossAttModel(env_params, context_env_encoder_params, robot='7D')
        # Create the AR model
        with open(osp.join(ar_model_folder, 'ar_params.json'), 'r') as f:
            ar_params = json.load(f)
        self.ar_model = AutoRegressiveModel(**ar_params)

        # Load the parameters and set the model to eval
        checkpoint = torch.load(osp.join(ar_model_folder, 'best_model.pkl'))
        for model, state_dict in zip([self.context_env_encoder, self.ar_model], ['context_state', 'ar_model_state']):
            model.load_state_dict(checkpoint[state_dict])
            model.eval()
            model.to(device)
        
        self.server = rospy.Service('distribution_sequence_predict', GetSamplingDistributionSequence, self.handle_request)

    def handle_request(self, req):
        print("receive req")

        pointcloud_array = np.array(req.obstacle_pointcloud)

        # # set the pointcloud into tensor
        map_data = tg_data.Data(pos=torch.as_tensor(pointcloud_array.reshape(-1, 3), dtype=torch.float, device=device))
        
        # normalize the start and goal configuration and set them into tensors
        tmp = (np.array([req.start_configuration, req.goal_configuration])+np.pi)%(2*np.pi)
        tmp[tmp<0] = tmp[tmp<0] + 2*np.pi
        tmp[tmp>0] = tmp[tmp>0] - np.pi
        path = (tmp-fu.q_min)/(fu.q_max-fu.q_min)

        search_dist_mu, search_dist_sigma, _ = get_search_dist(path, map_data, self.context_env_encoder, self.decoder_model, self.ar_model, self.quantizer_model, self.num_keys)
        
        # unnormalize the result.
        scaled_search_dist_mu =search_dist_mu*(fu.q_max-fu.q_min) + fu.q_min
        scaled_search_dist_sigma = search_dist_sigma@np.diag((fu.q_max-fu.q_min)[0]**2)

        result = GetSamplingDistributionSequenceResponse()

        for mean, sigma in zip(scaled_search_dist_mu, scaled_search_dist_sigma):
            sd = SamplingDistribution()
            sd.distribution_mean = mean.tolist()
            sd.distribution_convariance = sum(sigma.tolist(),[])
            result.distribution_sequence.append(sd)

        return result

if __name__ == "__main__":

    # ======================== Model loading :END ========================================
    # ======================== Data loading : START ======================================
    '''

    map_file = osp.join(val_data_folder, f'env_{env_num:06d}/map_{env_num}.ply')
    data_PC = o3d.io.read_point_cloud(map_file, format='ply')
    depth_points = np.array(data_PC.points) # <- Put point cloud data here.
    map_data = tg_data.Data(pos=torch.as_tensor(depth_points, dtype=torch.float, device=device))

    path_file = osp.join(val_data_folder, f'env_{env_num:06d}/path_{path_num}.p')
    data = pickle.load(open(path_file, 'rb'), encoding='latin1')
    # Normalize path
    tmp = (data['path']+np.pi)%(2*np.pi) # <- Give the start and goal points here 
    tmp[tmp<0] = tmp[tmp<0] + 2*np.pi
    tmp[tmp>0] = tmp[tmp>0] - np.pi
    path = (tmp-fu.q_min)/(fu.q_max-fu.q_min)
    # ======================== Data loading : END ======================================
    # ========================= INFERENCE : START ========================================
    # Generate search distribution
    search_dist_mu, search_dist_sigma, _ = get_search_dist(path, map_data, context_env_encoder, decoder_model, ar_model, quantizer_model, num_keys)
    print(search_dist_mu)
    print(search_dist_sigma)
    # ========================= INFERENCE : END ========================================
    '''
    rospy.init_node('distribution_sequence_prediction_server', anonymous=True)
    
    try:
        server = DistributionSequencePredServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
