''' A python script for saving data for 7D manipulation.
'''

import os
import sys

import pybullet as pyb
from os import path as osp
import pickle

# Set up path planning
from ompl import base as ob
from ompl import geometric as og

import panda_utils as pu
import argparse


def set_env(client_obj, env_num):
    '''
    Sets up the environment for the scene.
    param client_obj: bc.BulletClient object.
    :returns tuple: robot1ids, robot2ids, obstacles ids
    '''
    pyb.resetSimulation()
    robotid = pu.set_robot(client_obj)
    all_obstacles = pu.set_obstacles(client_obj, env_num, 8, 8)
    return robotid, all_obstacles


def generate_trajectories(client_obj, env_num, space, num_paths, file_dir, cur_path=0):
    '''Generate trajectories for the given environment.
    :param env_num: numpy seed to set.
    :param space: an ompl.base object
    '''
    robotid, all_obstacles = set_env(client_obj, env_num)
    si = ob.SpaceInformation(space)
    # Collect trajectories without obstacles
    valid_checker_obj = pu.ValidityCheckerDistance(
        client_obj,
        si,
        robotID=robotid[0],
        joints=robotid[1],
        obstacles=None
    )
    # # Collect trajectories with obstacles.
    # valid_checker_obj = pu.ValidityCheckerDistance(
    #     client_obj,
    #     si,
    #     robotID=robotid[0],
    #     joints=robotid[1],
    #     obstacles=all_obstacles
    # )
    # Run a simple planner:
    si.setStateValidityChecker(valid_checker_obj)

    start_state = ob.State(space)
    goal_state = ob.State(space)

    while cur_path<num_paths:
        start_state.random()
        while not valid_checker_obj.isValid(start_state()):
            start_state.random()

        goal_state.random()
        while not valid_checker_obj.isValid(goal_state()):
            goal_state.random()

        # Plan Path
        path, path_interpolated, success = pu.get_path(start_state, goal_state, si, 90)

        if success:
            print(f"Collected Path {cur_path} for  Env {env_num}")
            traj_data = {'path':path, 'path_interpolated':path_interpolated, 'success':success}
            pickle.dump(traj_data, open(osp.join(file_dir, f'path_{cur_path}.p'), 'wb'))
            cur_path += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', help="Folder where data is saved")
    parser.add_argument('--start', help="Start of the sampling", type=int)
    parser.add_argument('--samples', help="Number of samples to collect", type=int)
    parser.add_argument('--num_paths', help="Number of paths to collect for each environment", type=int)

    args = parser.parse_args()
    p = pu.get_pybullet_server('direct')

    space = ob.RealVectorStateSpace(7)
    bounds = ob.RealVectorBounds(7)
    
    # Set joint limits
    for i in range(7):
        bounds.setHigh(i, pu.q_max[0, i])
        bounds.setLow(i, pu.q_min[0, i])
    space.setBounds(bounds)

    for env_num in range(args.start, args.start+args.samples):
        env_file_dir = osp.join(args.log_dir, f'env_{env_num:06}')
        if not osp.isdir(env_file_dir):
            os.mkdir(env_file_dir)

        # Check if environment folder has enough trajectories.
        cur_path= len([filei for filei in os.listdir(env_file_dir) if filei.endswith('.p')])
        if cur_path==args.num_paths:
            continue
        generate_trajectories(p, env_num, space, args.num_paths, env_file_dir, cur_path)