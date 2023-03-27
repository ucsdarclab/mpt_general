''' Visualize the distributions of learned joint configurations.
'''
import torch
import pybullet as pyb
import numpy as np
from os import path as osp
import json
from torch.distributions import MultivariateNormal

import panda_utils as pdu
from modules.quantizers import VectorQuantizer
from modules.decoder import DecoderPreNormGeneral

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":

    # p = pdu.get_pybullet_server('gui')
    
    # define the quantization model.
    d_model = 512
    num_keys = 2048
    goal_index = num_keys + 1
    quantizer_model = VectorQuantizer(n_e=num_keys, e_dim=8, latent_dim=d_model)

    # Load quantizer model.
    dictionary_model_folder = '/root/data/general_mpt/model6'
    with open(osp.join(dictionary_model_folder, 'model_params.json'), 'r') as f:
        dictionary_model_params = json.load(f)

    decoder_model = DecoderPreNormGeneral(
        e_dim=dictionary_model_params['d_model'], 
        h_dim=dictionary_model_params['d_inner'], 
        c_space_dim=dictionary_model_params['c_space_dim']
    )
    # Load the quantization model.
    checkpoint = torch.load(osp.join(dictionary_model_folder, 'best_model.pkl'))
    
    # Load model parameters and set it to eval
    for model, state_dict in zip([quantizer_model, decoder_model], ['quantizer_state', 'decoder_state']):
        model.load_state_dict(checkpoint[state_dict])
        model.eval()
        model.to(device)
    
    # TODO: Set up the robot at sampled locations.
    # random_index = np.random.randint(0, num_keys)
    p = pdu.get_pybullet_server('gui')
    for random_index, rgba in zip([9, 730, 964], [[1, 0, 0, 0.4], [0, 1, 0, 0.4], [0, 0, 1, 0.4]]):
    # for random_index, rgba in zip([np.random.randint(0, num_keys)], [[1, 0, 0, 0.4], [0, 1, 0, 0.4], [0, 0, 1, 0.4]]):
        print(f"Using Index: {random_index}")

        quant_vector = quantizer_model.embedding(torch.tensor(random_index, device=device))
        quant_proj_vector = quantizer_model.output_linear_map(quant_vector)

        dist_mu, dist_simga = decoder_model(quant_proj_vector[None, None, :])
        dist_mu  = dist_mu.cpu().detach().squeeze()
        dist_sigma = dist_simga.cpu().detach().squeeze()

        search_dist_mu = torch.zeros((1, 7))
        search_dist_mu[0, :6] = dist_mu
        search_dist_sigma = torch.diag_embed(torch.ones((1, 7)))
        search_dist_sigma[0, :6, :6] = dist_sigma

        X = MultivariateNormal(search_dist_mu, search_dist_sigma)
        scale_pose = lambda x:  (x*(pdu.q_max - pdu.q_min)+pdu.q_min)[0]
        for _ in range(4):
            tmp_pose = scale_pose(X.sample())
            tmp_pose[-1] = 0.0
            pdu.set_robot_vis(p, tmp_pose, rgba)