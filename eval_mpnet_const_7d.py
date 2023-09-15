''' Constraint planning using MPNet and projecting points on the constraint surface.
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
from mpnet_models import MLP, Encoder

import panda_utils as pu
import eval_const_7d as ec7
import interactive_panda_kitchen as ipk
import interactive_kitchen_dev as ikd
import panda_constraint_shelf as pcs
import eval_mpnet_6d as em6

def scale_state(state):
    '''
    Scales trajectory from [-1, 1] back to q_min:q_max
    '''
    return (pu.q_max-pu.q_min)*(state+1)/2 + pu.q_min

def get_predict_points(start_tensor, goal_tensor, h, mlp_model, space, validity_checker, constraint_function, max_pred_steps):
    ''' Generate points connecting the start and goal tensor.
    '''
    is_connected = False
    cur_state = start_tensor[None, :]
    end_state = goal_tensor[None, :]
    forward_pred = [scale_state(cur_state.numpy())]
    reverse_pred = [scale_state(end_state.numpy())]
    forward = True
    for _ in range(max_pred_steps):
        if forward:
            mlp_input = torch.cat((cur_state, end_state, h), dim=1)
            next_state = mlp_model(mlp_input).clip(torch.ones(7)*-1, torch.ones(7)*1)
            scaled_state = scale_state(next_state.detach().numpy()).squeeze()
            # Project state
            constraint_function.project(scaled_state)
            while not validity_checker.isValid(scaled_state):
                next_state = mlp_model(mlp_input).clip(torch.ones(7)*-1, torch.ones(7)*1)
                scaled_state = scale_state(next_state.detach().numpy()).squeeze()
                # Project state
                constraint_function.project(scaled_state)
            # Check if we can connect the predict state with end state
            local_traj = em6.construct_traj(scaled_state, reverse_pred[0])
            # If valid path exists then return the points.
            if em6.valid_local_traj(local_traj, space, validity_checker):
                is_connected = True
                forward_pred.append(scaled_state[None, :])
                print("Connected")
                break
            forward_pred.append(scaled_state[None, :])
            cur_state = next_state
            forward=False
        else:
            mlp_input = torch.cat((end_state, cur_state, h), dim=1)
            next_state = mlp_model(mlp_input).clip(torch.ones(7)*-1, torch.ones(7)*1)
            scaled_state = scale_state(next_state.detach().numpy()).squeeze()
            # Project state
            constraint_function.project(scaled_state)
            while not validity_checker.isValid(scaled_state):
                next_state = mlp_model(mlp_input)
                scaled_state = scale_state(next_state.detach().numpy()).squeeze()
                # Project state
                constraint_function.project(scaled_state)
            local_traj = em6.construct_traj(forward_pred[-1], scaled_state)
            if em6.valid_local_traj(local_traj, space, validity_checker):
                is_connected = True
                reverse_pred.insert(0, scaled_state[None, :])
                print("Connected")
                break
            reverse_pred.insert(0, scaled_state[None, :])
            end_state = next_state
            forward=True
    return np.array(forward_pred+reverse_pred).squeeze(), is_connected


def mpnet_trajectory(start_q, goal_q, h, mlp_model, space, validity_checker, constraint_function):
    ''' Return mpnet trajectory using constraint planning.
    '''
    # Normalize trajectory points
    q = 2*(np.r_[start_q[None, :], goal_q[None, :]]-pu.q_min)/(pu.q_max-pu.q_min)-1
    q = torch.tensor(q, dtype=torch.float)
    success = False
    total_points_sampled = 0
    total_path_length = 0.0
    found_solution = False
    for _ in range(10):
        pred_points, success = get_predict_points(
                                            q[0], 
                                            q[-1], 
                                            h,
                                            mlp_model,
                                            space,
                                            validity_checker,
                                            constraint_function, 
                                            max_pred_steps=10
                                            )
        total_points_sampled+=pred_points.shape[0]
        total_plan_time = 0.0
        if success:
            # Find if we can plan consecutive points
            overall_path = []
            for i, _ in enumerate(pred_points[:-1, :]):
                sub_traj, path_length, plan_time, _, success = ikd.get_constraint_path_v2(
                    pred_points[i],
                    pred_points[i+1],
                    validity_checker_obj,
                    constraint_function,
                    plan_time=5
                )
                total_path_length += path_length
                total_plan_time += plan_time
                if not success:
                    break
                overall_path.append(sub_traj)
            found_solution = success
        if found_solution:
            break
    if found_solution:
        return np.concatenate(overall_path, axis=0), total_points_sampled, total_plan_time, total_path_length, True
    return np.r_[start_q[None, :], goal_q[None, :]], total_points_sampled, total_plan_time, total_path_length, False

if __name__ == "__main__":
    # Server for collision checking
    p_collision = pu.get_pybullet_server('direct')
    # Server for collision checking
    p_pcd = pu.get_pybullet_server('direct')
    # Server for visualization/execution
    p = pu.get_pybullet_server('direct')

    p.setAdditionalSearchPath(osp.join(os.getcwd(), 'assets'))

    # ========================== Load Model =======================
    enc_input_size=4120*3
    enc_output_size = 60
    mlp_output_size = 7

    mlp_model = MLP(enc_output_size+mlp_output_size*2, mlp_output_size)
    encoder_model = Encoder(enc_input_size, enc_output_size)

    #### Load the model

    model_file = '/root/data/mpnet_7d/model2/'
    epoch_num = 199

    checkpoint = torch.load(osp.join(model_file, f'model_{epoch_num}.pkl'))
    mlp_model.load_state_dict(checkpoint['mlp_state'])
    encoder_model.load_state_dict(checkpoint['encoder_state'])

    # ======================= End: Load Model ======================

    # ============================ Plan ============================
    # Load the constraint start and goal positions
    env_log_dir = '/root/data/panda_constraint/val_can_kitchen'
    run_data = []
    # env_num = 2
    for env_num in range(150):
        print("Planning for :", env_num)
        # Check if objects can be reached at the sampled point.
        if not osp.isfile(osp.join(env_log_dir, f'env_{env_num:06d}', 'table_target_q.pkl')):
            continue
        timing_dict = {}
        # Set up environment 
        print("Resetting Simulation")
        for client_id in [p, p_pcd, p_collision]:
            client_id.resetSimulation()

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
        with open('q_start_c.pkl', 'rb') as f:
            data = pickle.load(f)
            can_start_q = data['q_start']

        with open(osp.join(env_log_dir, f'env_{env_num:06d}', 'table_target_q.pkl'), 'rb') as f:
            data = pickle.load(f)
            can_goal_q = data['q_goal']

        ipk.panda_reset_open_gripper(p, pandaID, gripper_dist=0.1)
        ipk.panda_reset_open_gripper(p_collision, pandaID_col, gripper_dist=0.1)

        # ============== Get PCD ===========
        pcd = ipk.get_pcd(p_pcd)
        num_points = enc_input_size//3
        cropped_pcd = pcd.random_down_sample(num_points/np.asarray(pcd.points).shape[0])
        depth_points = np.asarray(cropped_pcd.points)

        # ==================== Define TSR and sample goal ==================
        goal_region = ikd.TSR(
            si,
            validity_checker_obj, 
            world_T_can, 
            offset_T_grasp=np.c_[np.eye(4, 3), np.array([-0.11, 0.0, 0.05, 1])]@can_T_ee, 
            goal_tolerance=np.array([0.0, 0.0, 0.5*np.pi])
        )
        test_state = ob.State(space)
        goal_samples = []
        for _ in range(5):
            goal_region.sampleGoal(test_state)
            sample_goal = np.array([test_state[i] for i in range(7)])
            goal_samples.append(sample_goal)
        goal_samples = np.r_[goal_samples]
        # Get encoded latent variables
        pc_data = torch.tensor(depth_points.reshape(-1)[None, :], dtype=torch.float)
        h = encoder_model(pc_data)

        # Plan MPNet path
        timing_dict['patch_time'] = 0.0
        timing_dict['plan_time'] = 0.0
        timing_dict['num_vertices'] = 0.0
        for can_goal_q_i in goal_samples:
            with ipk.Timer() as timer:
                path_can, num_vertices, plan_time, path_length, success = mpnet_trajectory(
                    can_start_q, 
                    can_goal_q, 
                    h, 
                    mlp_model,
                    space,
                    validity_checker_obj,
                    constraint_function
                    )
            timing_dict['num_vertices'] += num_vertices
            timing_dict['plan_time'] += timer()
            if success:
                break
            if timing_dict['plan_time']>50:
                break
        timing_dict['path_length'] = path_length
        timing_dict['success'] = success
        timing_dict['env_num'] = env_num
        run_data.append(timing_dict)

    result_log_folder = '/root/data/panda_constraint'
    with open(osp.join(result_log_folder, 'kitchen_timing_const_mpnet.pkl'), 'wb') as f:
        pickle.dump(run_data, f)