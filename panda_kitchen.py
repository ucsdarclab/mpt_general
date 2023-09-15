''' Kitchen interactive enviornment
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

import torch
from torch.distributions import MultivariateNormal
import torch_geometric.data as tg_data

import open3d as o3d
import json
import spatialmath as smp

from ompl_utils import get_ompl_state, get_numpy_state

# VQ-MPT model 
from modules.quantizers import VectorQuantizer
from modules.decoder import DecoderPreNormGeneral
from modules.encoder import EncoderPreNorm
from modules.autoregressive import AutoRegressiveModel, EnvContextCrossAttModel

import eval_const_7d as ec7
import interactive_panda_kitchen as ipk
import interactive_kitchen_dev as ikd
import panda_constraint_shelf as pcs


if __name__=="__main__":
    use_model = True
    state_space = "PJ"
    latent_project = False

    # Server for collision checking
    p_collision = pu.get_pybullet_server('direct')
    # Server for collision checking
    p_pcd = pu.get_pybullet_server('direct')
    # Server for visualization/execution
    p = pu.get_pybullet_server('direct')

    p.setAdditionalSearchPath(osp.join(os.getcwd(), 'assets'))

    # ============== Load VQ-MPT Model ======================
    dict_model_folder = '/root/data/general_mpt_panda_7d/model1'
    ar_model_folder = '/root/data/general_mpt_panda_7d/stage2/model1'
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    # Define the models
    d_model = 512
    # Get the number of keys from the saved data
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

    log_dir = '/root/data/panda_constraint/val_can_kitchen'
    run_data = []
    for env_num in range(150):
        print("Planning for :", env_num)
        # Check if objects can be reached at the sampled point.
        env_log_dir = osp.join(log_dir, f'env_{env_num:06d}')
        if not osp.isfile(osp.join(env_log_dir, 'table_target_q.pkl')):
            continue

        print("Resetting Simulation")
        for client_id in [p, p_pcd, p_collision]:
            client_id.resetSimulation()
        
        timing_dict = {}
        # Set up environment for simulation
        all_obstacles, itm_id = ipk.set_env(p, seed=env_num)
        kitchen = all_obstacles[0]
        # Set up environment for collision checking
        all_obstacles_coll, itm_id_coll = ipk.set_env(p_collision)
        # Set up environment for capturing pcd
        all_obstacles_pcd, itm_id_pcd = ipk.set_env(p_pcd)

        # Load the interactive robot
        pandaID, jointsID, fingerID = pu.set_robot(p)
        # Load the collision checking robot
        pandaID_col, jointsID_col, fingerID_col = pu.set_robot(p_collision)

        # Open the shelf
        shelf_index = 29
        for client_id, kitchen_id in zip([p, p_collision, p_pcd], [all_obstacles[0], all_obstacles_coll[0], all_obstacles_pcd[0]]):
            client_id.resetJointState(kitchen_id, shelf_index-2, -1.57)

        can_T_ee = np.array([[0., 0., 1, 0.], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        # Offset by 10cm along x-axis for allowing grasping and 5 cm along z-axis for avoiding 
        # collision w/ table.
        can_pose = np.array(p.getBasePositionAndOrientation(itm_id)[0])
        world_T_can = smp.base.rt2tr(np.eye(3), can_pose)

        # ============== Constraint trajectory planning =================
        # Find a constraint path without tilting the object
        tolerance = np.array([2*np.pi, 0.1, 0.1])
        constraint_function = pcs.EndEffectorConstraint(can_T_ee[:3, :3], tolerance, pandaID, jointsID)
        # Define OMPL parameters
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
        door_link_index = 29

        # Load the constraint start and goal positions
        with open('q_start_c.pkl', 'rb') as f:
            data = pickle.load(f)
            can_start_q = data['q_start']
        
        with open(osp.join(env_log_dir, 'table_target_q.pkl'), 'rb') as f:
            data = pickle.load(f)
            can_goal_q = data['q_goal']
        
        ipk.panda_reset_open_gripper(p, pandaID, gripper_dist=0.1)
        ipk.panda_reset_open_gripper(p_collision, pandaID_col, gripper_dist=0.1)

        # ============== Get PCD ===========
        pcd = ipk.get_pcd(p_pcd)
        map_data = tg_data.Data(pos=torch.as_tensor(np.asarray(pcd.points), dtype=torch.float, device=device))

        # Sample multiple goals
        if use_model:
            # Define TSR for sampling goal poses for inputing to network
            goal_region = ikd.TSR(
                si,
                validity_checker_obj, 
                world_T_can, 
                offset_T_grasp=np.c_[np.eye(4, 3), np.array([-0.11, 0.0, 0.05, 1])]@can_T_ee, 
                goal_tolerance=np.array([0.0, 0.0, 0.5*np.pi])
            )
            test_state = ob.State(space)
            goal_samples = []
            for _ in range(20):
                goal_region.sampleGoal(test_state)
                sample_goal = np.array([test_state[i] for i in range(7)])
                goal_samples.append(sample_goal)
            goal_samples = np.r_[goal_samples]

        
            timing_dict['patch_time'] = 0.0
            timing_dict['plan_time'] = 0.0
            timing_dict['num_vertices'] = 0.0
            for can_goal_q in goal_samples:
                with ipk.Timer() as timer:
                    # Single search
                    n_start_n_goal = (np.r_[can_start_q[None, :], can_goal_q[None, :]]-pu.q_min)/(pu.q_max-pu.q_min)
                    if latent_project:
                        search_dist_mu, search_dist_sigma, patch_time = ec7.get_search_proj_distv2(
                            n_start_n_goal,
                            can_start_q[None, :],
                            map_data,
                            context_env_encoder,
                            decoder_model,
                            ar_model,
                            quantizer_model,
                            num_keys
                        )
                    else:
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
                    # # Multi-search
                    # n_start = (can_start_q[None, :]-pu.q_min)/(pu.q_max-pu.q_min)
                    # n_goals = (goal_samples-pu.q_min)/(pu.q_max-pu.q_min)
                    # search_dist_mu, search_dist_sigma, _ = ikd.get_search_dist_batch(
                    #     n_start,
                    #     n_goals,
                    #     map_data, 
                    #     context_env_encoder, 
                    #     decoder_model, 
                    #     ar_model, 
                    #     quantizer_model, 
                    #     num_keys
                    # )
                timing_dict['patch_time'] +=timer()
                print("Planning using v2")
                traj_cupboard, path_length, plan_time, num_vertices , success = ikd.get_constraint_path_v2(
                    can_start_q, 
                    can_goal_q, 
                    validity_checker_obj, 
                    constraint_function, 
                    search_dist_mu, 
                    search_dist_sigma,
                    plan_time=2.5
                )
                timing_dict['path_length'] = path_length
                timing_dict['plan_time']+=plan_time
                timing_dict['num_vertices']+=num_vertices
                if success:
                    break
        else:
            search_dist_mu, search_dist_sigma = None, None
            traj_cupboard, path_length, plan_time, num_vertices , success = ikd.get_constraint_path(
                            can_start_q, 
                            world_T_can, 
                            validity_checker_obj, 
                            constraint_function, 
                            search_dist_mu, 
                            search_dist_sigma,
                            plan_time=100,
                            state_space=state_space
                        )
            timing_dict['patch_time'] = 0.0
            timing_dict['plan_time'] = plan_time
            timing_dict['path_length'] = path_length
            timing_dict['num_vertices']=num_vertices

        timing_dict['env_num'] = env_num
        timing_dict['success'] = success

        run_data.append(timing_dict)
    
    result_log_folder = '/root/data/panda_constraint'
    if use_model:
        if latent_project:
            file_name = f'kitchen_timing_const_vqmpt_proj_single_multi_sample_20.pkl'
        else:
            file_name = f'kitchen_timing_const_vqmpt_single_multi_sample_20.pkl'
    else:
        file_name = f'kitchen_timing_const_{state_space}.pkl'
    
    with open(osp.join(result_log_folder, file_name), 'wb') as f:
        pickle.dump(run_data, f)