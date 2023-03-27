''' Set up experiment for dual-arm shelf environment
'''

import os
import sys
# To make sure this can access all packages in the previous folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pybullet as pyb
import numpy as np

# Set up path planning
from ompl import base as ob
from ompl import geometric as og

import argparse
import pickle
from os import path as osp

import panda_utils as pdu
import dual_arm_utils as dau
import dual_arm_exp as dae
import collect_data as cd
import panda_shelf_env as pse
import eval_14d as e14d
from panda_utils import box_length, box_width, rgba, sph_radius

def try_target_location(
        client_obj, 
        robotID, 
        jointsID, 
        obstacles, 
        bp= np.array([0.18, 0.5, 0.2]),
        bo=[np.pi/2, -np.pi/2, np.pi]):
    ''' A function to try placing the end-effector randomly at a given end-effoctor position and
    orientation. 
    :param client_obj: pybullet scene object.
    :param robotID: pybullet id of robot to place in sim.
    :param jointsID: pybullet ids of joint links.
    :param obstacles: list of obstacle ids to check collision with.
    :param bp: np.array of end-effector position.
    :param bo: orientation of the robot.

    '''
    set_joint_pose = np.array(pse.set_IK_position(client_obj, robotID, jointsID, bp, bo))[:7]
    # Check if the robot is in self-collision/collision w/ obstacles.
    if pse.check_self_collision(robotID) or pdu.get_distance(obstacles, robotID)<=0 or (not pse.check_bounds(set_joint_pose)):
        # If so randomize the joints and calculate IK once again.
        random_joint_pose = (pdu.q_min + (pdu.q_max - pdu.q_min)*np.random.rand(7))[0]
        pdu.set_position(robotID, jointsID, random_joint_pose)
        set_joint_pose = np.array(pse.set_IK_position(client_obj, robotID, jointsID, bp, bo))[:7]
        if pse.check_self_collision(robotID) or pdu.get_distance(obstacles, robotID)<=0 or (not pse.check_bounds(set_joint_pose)):
            # if pse.check_self_collision(robotID):
            #     # print("Robot is in collision with itself!!")
            # if not pse.check_bounds(set_joint_pose):
            #     print("Joints out of bounds")
            # if pdu.get_distance(obstacles, robotID)<=0:
            #     print("Robot is in collision with obstacles.")
            return False, set_joint_pose
    return True, set_joint_pose


def generate_scene(client_obj):
    '''
    Set up the scene for planning.
    :param client_obj: pybullet client object.
    :return list: ids of objects.
    '''
    # Place the shelf.
    all_obstacles = pse.place_shelf(client_obj, base_pose=[0.35, 0.85, 0.1], base_orient=[np.pi/2, 0, np.pi])

    rbga_wood = [0.54, 0.31, 0.21, 1]
    # Define table/box
    base_dim = [0.2, 0.3, 0.1]
    geomBox = client_obj.createCollisionShape(pyb.GEOM_BOX, halfExtents=base_dim)
    visualBox = client_obj.createVisualShape(pyb.GEOM_BOX, halfExtents=base_dim, rgbaColor=rbga_wood)
    all_obstacles.append(client_obj.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=geomBox,
            baseVisualShapeIndex=visualBox,
            basePosition=np.r_[0.8, 0, base_dim[2]],
            baseOrientation=pyb.getQuaternionFromEuler([0.0, 0.0, 0])
        )
    )
    # Define shelf
    base_dim = [0.15, 0.35, 0.025]
    geomBox = client_obj.createCollisionShape(pyb.GEOM_BOX, halfExtents=base_dim)
    visualBox = client_obj.createVisualShape(pyb.GEOM_BOX, halfExtents=base_dim, rgbaColor=rbga_wood)
    all_obstacles.append(client_obj.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=geomBox,
            baseVisualShapeIndex=visualBox,
            basePosition=np.r_[-1, 0, 0.5],
            baseOrientation=pyb.getQuaternionFromEuler([0.0, 0.0, 0])
        )
    )
    return all_obstacles

# Dictionary of poses
ee_poses = {
    's': {
        'p': np.r_[-0.1, 0.5, 0.2], 
        'o': np.r_[np.pi/2, -np.pi/2, np.pi], 
        's': np.r_[0.3, 0.0, 0.5]
    },
    't': {
        'p': np.r_[0.7, -0.2, 0.35],
        'o': np.r_[np.pi, 0., 0.0],
        's': np.r_[0.3, 0.5, 0.0]
    },
    'fs': {
        'p': np.r_[-0.8, -0.2, 0.6], 
        'o': np.r_[0, -np.pi/2, 0],
        's': np.r_[0, 0.4, 0.0]
    }
}

def get_start_n_goal(robotid1, robotid2, obstacles, seq):
    '''
    Get start and goal states.
    :param robotid1:
    :param robotid2:
    :param obstacles:
    :param seq:
    :returns list: the start and goal position of combined robot.
    '''
    # Place the robot in a random start and goal position.
    start_goal_pose = []
    for i in range(2):
        robot1_pose = ee_poses[seq[0][i]]
        bp_robot_1 = robot1_pose['p'] + robot1_pose['s']*np.random.rand(3)
        success1, set_joint_pose = try_target_location(p, robotid1[0], robotid1[1], obstacles, bp_robot_1, robot1_pose['o'])
        count = 0
        while not success1 and count<10:
            success1, set_joint_pose = try_target_location(p, robotid1[0], robotid1[1], obstacles, bp_robot_1, robot1_pose['o'])
            count += 1
        # print(pse.get_robot_end_effector_pose(p, robotid1[0]))

        # Robot 2 - start and goal.
        robot2_pose = ee_poses[seq[1][i]]
        bp_pose_2 = robot2_pose['p'] + robot2_pose['s']*np.random.rand(3)
        success2, set_joint_pose_2 = try_target_location(p, robotid2[0], robotid2[1], obstacles, bp=bp_pose_2, bo=robot2_pose['o'])
        count = 0
        while not success2 and count<15:
            success2, set_joint_pose_2 = try_target_location(p, robotid2[0], robotid2[1], obstacles, bp=bp_pose_2, bo=robot2_pose['o'])
            count += 1
        # print(pse.get_robot_end_effector_pose(p, robotid2[0]))
        if not (success1 and success2):
            return None
        start_goal_pose.append(np.r_[pse.get_joint_position(robotid1[0], robotid1[1]), pse.get_joint_position(robotid2[0], robotid2[1])])
    return start_goal_pose


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', help="Folder where data is saved")
    parser.add_argument('--start', help="Start of the paths", type=int)
    parser.add_argument('--num_paths', help="Number of paths to collect", type=int)

    args = parser.parse_args()

    # Create a directory to store data.
    env_num = 0
    env_file_dir = osp.join(args.log_dir, f'env_{env_num:06d}')
    if not osp.isdir(env_file_dir):
        os.mkdir(env_file_dir)
    # Check if environment folder has enough trajectories.
    cur_path = len([filei for filei in os.listdir(env_file_dir) if filei.endswith('.p')])

    p = pdu.get_pybullet_server('direct')
    robotid1, robotid2 = dau.set_dual_robot(p)
    seed = 1

    # Set the scene up
    all_obstacles = generate_scene(p)
    seq = [
        [['s', 't'], ['fs', 's']], 
        [['t', 's'], ['s', 'fs']],
        [['t', 's'], ['fs', 's']],
        [['s', 't'], ['s', 'fs']]
    ]

    space = ob.RealVectorStateSpace(14)
    bounds = ob.RealVectorBounds(14)
    
    # Set joint limits
    for i in range(7):
        bounds.setHigh(i, pdu.q_max[0, i])
        bounds.setHigh(i+7, pdu.q_max[0, i])
        bounds.setLow(i, pdu.q_min[0, i])
        bounds.setLow(i+7, pdu.q_min[0, i])
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)
    validity_checker_obj = dau.ValidityCheckerDualDistance(
        si,
        robotID_1=(robotid1[0], robotid1[1]),
        robotID_2=(robotid2[0], robotid2[1]),
        obstacles=all_obstacles
    )
    si.setStateValidityChecker(validity_checker_obj)
    
    while cur_path<args.num_paths:
        start_n_goal = None
        # Set up planning for the robots.
        while start_n_goal is None:
            # Randomly choose one of the sequence.
            start_n_goal = get_start_n_goal(robotid1, robotid2, all_obstacles, seq[np.random.randint(0, 4)])

        # Plan for the given start and goal states.
        start_state = e14d.get_ompl_state(space, start_n_goal[0])
        goal_state = e14d.get_ompl_state(space, start_n_goal[-1])
        
        # Save data
        path, path_interpolated, success = dau.get_path(start_state, goal_state, si, total_time=300)
        if success:
            print(f"Collected path {cur_path}")
            traj_data = {'path':path, 'path_interpolated': path_interpolated, 'success': success}
            pickle.dump(traj_data, open(osp.join(env_file_dir, f'path_{cur_path}.p'), 'wb'))
            cur_path +=1
