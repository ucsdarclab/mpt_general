''' A python script for saving data for bi-manual manipulation.
'''

import os
import sys
# To make sure this can access all packages in the previous folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pybullet as pyb
from os import path as osp
import pickle

# Set up path planning
from ompl import base as ob
from ompl import geometric as og

import panda_utils as pu
import dual_arm_utils as dau

import argparse


def set_env(client_obj, env_num):
    '''
    Sets up the environment for the scene.
    param client_obj: bc.BulletClient object.
    :returns tuple: robot1ids, robot2ids, obstacles ids
    '''
    pyb.resetSimulation()
    robotid1, robotid2 = dau.set_dual_robot(client_obj)
    all_obstacles = dau.set_obstacles(client_obj, env_num, 8, 8, robotid1[0], robotid2[0])
    return robotid1, robotid2, all_obstacles


def generate_trajectories(client_obj, env_num, space, num_paths, file_dir):
    '''Generate trajectories for the given environment.
    :param env_num: numpy seed to set.
    :param space: an ompl.base object
    '''
    robotid1, robotid2, all_obstacles = set_env(client_obj, env_num)
    si = ob.SpaceInformation(space)
    valid_checker_obj = dau.ValidityCheckerDualDistance(
        si,
        robotID_1=(robotid1[0], robotid1[1]),
        robotID_2=(robotid2[0], robotid2[1]),
        obstacles=all_obstacles
    )
    # Run a simple planner:
    si.setStateValidityChecker(valid_checker_obj)

    start_state = ob.State(space)
    goal_state = ob.State(space)

    cur_path = 0
    while cur_path<num_paths:
        start_state.random()
        while not valid_checker_obj.isValid(start_state()):
            start_state.random()

        goal_state.random()
        while not valid_checker_obj.isValid(goal_state()):
            goal_state.random()

        # Plan Path
        path, path_interpolated, success = dau.get_path(start_state, goal_state, si, 90)

        if success:
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

    space = ob.RealVectorStateSpace(14)
    bounds = ob.RealVectorBounds(14)
    
    # Set joint limits
    for i in range(7):
        bounds.setHigh(i, pu.q_max[0, i])
        bounds.setHigh(i+7, pu.q_max[0, i])
        bounds.setLow(i, pu.q_min[0, i])
        bounds.setLow(i+7, pu.q_min[0, i])
    space.setBounds(bounds)

    for env_num in range(args.start, args.start+args.samples):
        env_file_dir = osp.join(args.log_dir, f'env_{env_num:06}')
        if not osp.isdir(env_file_dir):
            os.mkdir(env_file_dir)

        generate_trajectories(p, env_num, space, args.num_paths, env_file_dir)