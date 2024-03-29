''' Visualize robot trajectories
'''

import pybullet as pyb
import pybullet_data
import pybullet_utils.bullet_client as bc

import numpy as np
import pickle
import os
from os import path as osp

# Set up path planning
from ompl import base as ob
from ompl import geometric as og

import time
import argparse

from panda_utils import q_max, q_min, get_pybullet_server
from panda_utils import set_simulation_env, set_obstacles, set_robot
from panda_utils import set_robot_vis, set_position
import panda_shelf_env as pse


def set_visual_env(client_obj, num_boxes, num_spheres, seed):
    '''
    Generate environment with randomly placed obstacles in space.
    :param client_obj: bc.BulletClient object
    :param num_boxes:
    :param num_spheres:
    :param seed:
    :returns ValidityCheckerObj:
    '''
    set_simulation_env(client_obj)
    panda, joints, _ = set_robot(client_obj)
    obstacles = set_obstacles(client_obj, num_boxes=num_boxes, num_spheres=num_spheres, seed = seed)
    return panda, joints, obstacles

def set_visual_shelf_env(client_obj, seed):
    '''
    Generate environment with shelf and objects.
    :param client_obj: bc.BulletClient object.
    :param seed: random seed used to generate enviornment.
    '''
    set_simulation_env(client_obj)
    panda, joints, _ = set_robot(client_obj)
    obstacles = pse.place_shelf_and_obstacles(client_obj, seed)
    return panda, joints, obstacles


def set_visual_env_no_robot(client_obj, num_boxes, num_spheres, seed):
    '''
    Generate environment with randomly placed obstacles in space.
    :param client_obj: bc.BulletClient object
    :param num_boxes:
    :param num_spheres:
    :param seed:
    :returns ValidityCheckerObj:
    '''
    set_simulation_env(client_obj)
    obstacles = set_obstacles(client_obj, num_boxes=num_boxes, num_spheres=num_spheres, seed = seed)
    return obstacles

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', help='Train/Predicted', choices=['train', 'predict'])
    parser.add_argument('--planner', help='Planner', choices=['rrt', 'rrtstar'])
    parser.add_argument('--env_num', help='int b/w 2000-2500', type=int)
    parser.add_argument('--shelf', help='If true, use the shelf environment', action='store_true')
    args = parser.parse_args()

    env_num = args.env_num
    path_num = 0

    p = get_pybullet_server('gui')

    # ==================================================================
    success = False
    # ================== Ground Truth Data ===============================
    if args.method == 'train':
        if args.shelf:
            print("Not implemented")
        else:
            dataFolder = "/root/data/panda_models/model0/rrtstar"
            index_num = env_num - 2000
            pathFile = osp.join(dataFolder, f"rrtstar_val_plan_{2000+(index_num//100)*100:06d}.p")
            step_size = 1
            with open(pathFile, "rb") as f:
                pathTraj = pickle.load(f)
            success = pathTraj[index_num%100]["success"]
            if success:
                path = pathTraj[index_num%100]['jointPath']
            else:
                path = None
    # ======================================================================
    # ================== Predicted Data ====================================
    if args.method == 'predict':
        model_folder = '/root/data/general_mpt/stage2/model8'
        planner_type = args.planner
        # step_size = 50
        step_size = 4
        if args.shelf:
            with open(osp.join(model_folder, f'eval_val_plan_{planner_type}_shelf_{2000:06d}.p'), 'rb') as f:
                eval_data = pickle.load(f)
        else:
            with open(osp.join(model_folder, f'eval_val_plan_{planner_type}_{2000:06d}.p'), 'rb') as f:
                eval_data = pickle.load(f)
        index_num = env_num - 2000
        path = eval_data['Path'][index_num]
        success = eval_data['Success'][index_num]
    # # ====================================================================
    # data_folder = '/root/data/pandav3/train'
    
    # # Define camera 
    # viewMatrix = p.computeViewMatrixFromYawPitchRoll(
    #     cameraTargetPosition=[0, 0, 0],
    #     distance = 2.2,
    #     yaw=0,
    #     pitch=-20.6,
    #     roll=-2.2,
    #     upAxisIndex=2
    # )
    # projectionMatrix = p.computeProjectionMatrixFOV(
    #     fov=55.0,
    #     aspect=1.0,
    #     nearVal=0.1,
    #     farVal=3.1
    # )
    if success:
        if args.shelf:
            # Shelf environment
            panda, joints, obstacles = set_visual_shelf_env(p, seed=env_num)
        else:
            # Random objects environment
            panda, joints, obstacles = set_visual_env(p, 6, 6, seed=env_num)

        # obstacles = set_visual_env_no_robot(p, 6, 6, seed=env_num)
        panda_start_vis = set_robot_vis(p, path[0], [0, 1, 0, 0.6])
        panda_goal_vis = set_robot_vis(p, path[-1], [1, 0, 0, 0.6])
        alpha = np.linspace(0, 1, step_size)[:, None]
        time.sleep(10)
        for i, _ in enumerate(path[:-1]):
            # Interpolate b/w joints.
            tmp_path = (1-alpha)*path[i] + alpha*path[i+1]
            for pos in tmp_path:
                # Set robot position.
                set_position(panda, joints, pos)
                # tmp_robot = set_robot_vis(p, pos, [1, 1, 1 , 0.6])
                # width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                #     width=480, 
                #     height=480,
                #     viewMatrix=viewMatrix,
                #     projectionMatrix=projectionMatrix
                # )
                time.sleep(0.1)
        time.sleep(1)
    else:
        print("Couldn't find a successful path")
