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
import collect_data as dau

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
    step_size = 50
    robotid1, robotid2, all_obstacles = dau.set_env(p, args.env_num)
    if args.method == 'train':
        dataFolder = "/root/data/bi_panda/train"
        with open(osp.join(dataFolder, f'env_{env_num:06d}', f'path_{path_num}.p'), 'rb') as f:
            pathTraj = pickle.load(f)
        path = pathTraj['path']

    alpha = np.linspace(0, 1, step_size)[:, None]
    for i, _ in enumerate(path[:-1]):
        tmp_path = (1-alpha)*path[i] + alpha*path[i+1]
        for pos in tmp_path:
            pdu.set_position(robotid1[0], robotid1[1], pos[:7])
            pdu.set_position(robotid2[0], robotid2[1], pos[7:])
            time.sleep(0.1)