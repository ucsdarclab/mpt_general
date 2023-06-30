''' Visualize dual arm.
'''

import os
import sys
# To make sure this can access all packages in the previous folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os.path as osp
import pybullet as pyb
import pickle
import argparse
import time

import numpy as np

import panda_utils as pdu
import collect_data as cd
import dual_arm_utils as dau
import dual_arm_shelf as das

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', help='Train/Predicted', choices=['train', 'predict'])
    parser.add_argument('--planner', help='Planner', choices=['rrt', 'rrtstar'])
    parser.add_argument('--env_num', help='int b/w 2000-2500', type=int)
    parser.add_argument('--path_num', help='path number', type=int)

    args = parser.parse_args()
    p = pdu.get_pybullet_server('gui')

    env_num = args.env_num
    path_num = args.path_num
    # step_size = 20
    step_size = 1

    # Shelf environment
    robotid1, robotid2 = dau.set_dual_robot(p)
    all_obstacles = das.generate_scene(p)
    # Shelf - Training Data
    # data_folder = '/root/data/bi_panda_shelf/val'
    # with open(osp.join(data_folder, f'env_{env_num:06d}', f'path_{path_num}.p'), 'rb') as f:
    #     path_data = pickle.load(f)
    #     path = path_data['path']

    # VQ-MPT - Eval Data
    data_folder = '/root/data/general_mpt_bi_panda/stage2/model1'
    with open(osp.join(data_folder, f'eval_val_plan_rrt_shelf_{0:06d}.p'), 'rb') as f:
        data = pickle.load(f)
    if data["Success"][path_num]:
        path = data['Path'][path_num]
    # robotid1, robotid2, all_obstacles = cd.set_env(p, args.env_num)
    # pyb.removeBody(robotid1[0])
    # pyb.removeBody(robotid2[0])
    # if args.method == 'train':
    #     dataFolder = "/root/data/bi_panda/train"
    #     with open(osp.join(dataFolder, f'env_{env_num:06d}', f'path_{path_num}.p'), 'rb') as f:
    #         pathTraj = pickle.load(f)
    #     path = pathTraj['path']
    #     success = True
    # elif args.method == 'predict':
    #     dataFolder="/root/data/general_mpt_bi_panda/stage2/model1"
    #     with open(osp.join(dataFolder, f'eval_val_plan_{args.planner}_{2001:06d}.p'), 'rb') as f:
    #         eval_data = pickle.load(f)
    #     index_num = env_num-2001
    #     path = eval_data['Path'][index_num]
    #     success = eval_data['Success'][index_num]

    # Set Robot starting and goal position for robot 1
    dau.set_dual_robot_vis(p, path[0], [1, 0 ,0, 0.7])
    dau.set_dual_robot_vis(p, path[-1], [0, 1 ,0, 0.7])

    # # # Visualize trajectories - good for images.
    # for i, _ in enumerate(path[0:-1]):
    #     step_size = int(1+np.linalg.norm(path[i+1]-path[i])//3)
    #     print(step_size)
    #     alpha = np.linspace(0, 1, step_size+1)[:, None]
    #     tmp_path = (1-alpha)*path[i] + alpha*path[i+1]
    #     for pos in tmp_path[:-1]:
    #         dau.set_dual_robot_vis(p, pos, [1, 1, 1, 0.6])

    # for i, pos in enumerate(path[0:-1]):
    #     dau.set_dual_robot_vis(p, pos, [1, 1, 1, 0.7])

    # Visualize traj - good for videos
    time.sleep(10)
    step_size = 20
    alpha = np.linspace(0, 1, step_size)[:, None]
    for i, _ in enumerate(path[:-1]):
        tmp_path = (1-alpha)*path[i] + alpha*path[i+1]
        for pos in tmp_path:
            pdu.set_position(p, robotid1[0], robotid1[1], pos[:7])
            pdu.set_position(p, robotid2[0], robotid2[1], pos[7:])
            time.sleep(0.25)
    time.sleep(1)