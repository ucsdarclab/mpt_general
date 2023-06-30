''' Contains useful functions to interface with the dual panda arms
'''

import os
import sys
import numpy as np
import pybullet as pyb

import pickle
from os import path as osp

import argparse

# To make sure this can access all packages in the previous folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up path planning
from ompl import base as ob
from ompl import geometric as og

from panda_utils import get_pybullet_server, set_robot, q_max, q_min, set_robot_vis
from panda_utils import set_position
from panda_utils import check_self_collision
from panda_utils import get_distance
from panda_utils import get_random_pos, box_length, sph_radius, rgba, box_width
import time


def set_dual_robot(clientID):
    ''' Sets up the two robot in the given server.
    :param clientID: a pybullet_utils.BulletClient object.
    :returns tuple: a tuple w/ 2 objects of (robotID, jointsID, fingerID) for each robot.
    '''
    robot1_ids = set_robot(
        clientID,
        base_pose=np.array([0.42, 0.0, 0.0]),
        base_orientation=np.array([0, 0, np.pi/2])
    )

    robot2_ids = set_robot(
        clientID,
        base_pose=np.array([-0.42, 0.0, 0.0]),
        base_orientation=np.array([0, 0, np.pi/2])
    )
    return (robot1_ids, robot2_ids)


def set_vis_dual_robot(clientID):
    ''' Sets up the two robot in the given server.
    :param clientID: a pybullet_utils.BulletClient object.
    :returns tuple: a tuple w/ 2 objects of (robotID, jointsID, fingerID) for each robot.
    '''
    robot1_ids = set_robot_vis(
        clientID,
        base_pose=np.array([0.42, 0.0, 0.0]),
        base_orientation=np.array([0, 0, np.pi/2])
    )

    robot2_ids = set_robot_vis(
        clientID,
        base_pose=np.array([-0.42, 0.0, 0.0]),
        base_orientation=np.array([0, 0, np.pi/2])
    )
    return (robot1_ids, robot2_ids)

def set_dual_robot_vis(client_obj, pose, rgbaColor):
    '''
    Set up dual robots for visualizing purposes.
    :param client_obj: a pybullet.client.BulletClient object.
    :param pose: a 14D vector for positioning robot1 and 2.
    :param rgbaColor: The color of both robots.
    '''
    # Spawn the robots.
    robot1_vis_ids, robot2_vis_ids = set_vis_dual_robot(client_obj)
    # Get the joint info
    numLinkJoints = pyb.getNumJoints(robot1_vis_ids[0])
    # Change the color of the robot.
    for j in range(numLinkJoints):
        client_obj.changeVisualShape(robot1_vis_ids[0], j, rgbaColor=rgbaColor)
        client_obj.changeVisualShape(robot2_vis_ids[0], j, rgbaColor=rgbaColor)
    
    #  Set the robot1 to a particular pose.
    set_position(client_obj, robot1_vis_ids[0], robot1_vis_ids[1], pose[:7])
    set_position(client_obj, robot2_vis_ids[0], robot2_vis_ids[1], pose[7:])

    return robot1_vis_ids, robot2_vis_ids


# Check collision with base
def check_robot_base_collision(obstacle, robotID):
    '''
    Return True is the robot base is in collision with the robot.
    :param obstacle: The pybullet id for the obstacle.
    :param robotID: pybullet id for the robot.
    :returns bool: True if the robot base is in collision.
    '''
    distance = min(
        link[8] for link in pyb.getClosestPoints(bodyA=obstacle, bodyB=robotID, distance=10)[:3]
    )
    return distance<0


def set_obstacles(client_obj, seed, num_boxes, num_spheres, robot_id1, robot_id2):
    ''' Sets up the obstacles in the environment.
    :param client_obj: A pybullet_utils.BulletClient object.
    :param seed: Random seed to be set.
    :param num_boxes: Number of boxes to be used.
    :param num_spheres: Number of spheres to be used.
    :return list: returns the ids of obstacles set in the simulation.
    '''
    # Define the box obstacle
    geomBox = client_obj.createCollisionShape(pyb.GEOM_BOX, halfExtents=[box_length/2, box_width/2, box_width/2])
    visualBox = client_obj.createVisualShape(pyb.GEOM_BOX, halfExtents=[box_length/2, box_width/2, box_width/2], rgbaColor=rgba)
    # Define the cylinderical obstacle
    geomCylinder = client_obj.createCollisionShape(pyb.GEOM_CYLINDER, radius=sph_radius, height= 0.1)
    visualCylinder = client_obj.createVisualShape(pyb.GEOM_CYLINDER, radius=sph_radius, length=0.1, rgbaColor=rgba)
    # Define the spherical obstacle
    geomSphere = client_obj.createCollisionShape(pyb.GEOM_SPHERE, radius=sph_radius)
    visualSphere = client_obj.createVisualShape(pyb.GEOM_SPHERE, radius=sph_radius, rgbaColor=rgba)

    np.random.seed(seed)
    robot_base_pose = [np.r_[0.42, 0., 0.], np.r_[-0.42, 0., 0.]]

    # Define square objects, position in spherical co-ordinates
    boxXYZ = get_random_pos(num_points=num_boxes)
    for i, j in enumerate(np.random.randint(2, size=num_boxes)):
        boxXYZ[i] = boxXYZ[i]+robot_base_pose[j]
    boxOri = np.random.rand(num_boxes)*np.pi*2

    offset = np.r_[0.0, 0.0, 0.5]
    obstacles_box = [
        client_obj.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=geomBox,
            baseVisualShapeIndex=visualBox,
            basePosition=boxXY_i+offset,
            baseOrientation=pyb.getQuaternionFromEuler([0.0, 0.0, boxOri_i])
        )
        for boxXY_i, boxOri_i in zip(boxXYZ, boxOri)
    ]
    # Check if the obstacles are in collision with the robot base.
    new_obstacles_box = [obs 
        for obs in obstacles_box 
            if not (check_robot_base_collision(obs, robot_id1) or check_robot_base_collision(obs, robot_id2))
    ]
    # Remove the obstacles from env
    for obs in obstacles_box:
        if obs not in new_obstacles_box:
            pyb.removeBody(obs)

    # Define spherical objects, position in spherical co-ordinates
    sphXYZ = get_random_pos(num_points=num_spheres)
    for i, j in enumerate(np.random.randint(2, size=num_spheres)):
        sphXYZ[i] = sphXYZ[i]+robot_base_pose[j]
    offset = np.r_[0.0, 0.0, 0.33]
    obstacles_sph = [
        client_obj.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=geomSphere,
            baseVisualShapeIndex=visualSphere,
            basePosition=sphXY_i+offset
        )
        for sphXY_i in sphXYZ
    ]
    # Check if the obstacles are in collision with the robot base.
    new_obstacles_sph = [ obs
        for obs in obstacles_sph
            if not (check_robot_base_collision(obs, robot_id1) or check_robot_base_collision(obs, robot_id2))
    ]
    for obs in obstacles_sph:
        if obs not in new_obstacles_sph:
            pyb.removeBody(obs)

    return new_obstacles_box+new_obstacles_sph

class ValidityCheckerDualDistance(ob.StateValidityChecker):
    ''' A class to check the validity of the state for bi-manual robot setup.
    '''
    defaultOrientation = pyb.getQuaternionFromEuler([0, 0, 0])
    def __init__(self, client_obj, si, robotID_1, robotID_2, obstacles=None):
        ''' initialize the class object.
        :param si: an object o type ompl.base.SpaceInformation
        :param robotID_1: A tuple of robot_ID, joints_ID for robot 1
        :param robotID_2: A tuple of robot_ID, joints_ID for robot 2
        :param obstacles: A list of obstacles ID
        '''
        super().__init__(si)
        self.client_obj = client_obj
        self.obstacles = obstacles
        self.robotID_1 = robotID_1
        self.robotID_2 = robotID_2

    def isValid(self, state):
        '''
        Check if the given state is valid. The expected state size is 14, 
        the first 7 values for robot 1 and the last 7 for robot 2
        :return bool: True if the state is valid.
        '''
        # Set robot position
        set_position(self.client_obj, self.robotID_1[0], self.robotID_1[1], [state[i] for i in range(7)])
        set_position(self.client_obj, self.robotID_2[0], self.robotID_2[1], [state[i] for i in range(7, 14)])
        # Check for self collision
        if check_self_collision(self.robotID_1[0]) or check_self_collision(self.robotID_2[0]):
            return False

        # Check for collision b/w robots
        if get_distance(self.client_obj, [self.robotID_1[0]], self.robotID_2[0])<0:
            return False

        # Check if either robots are colliding w/ obstacles.
        if self.obstacles is not None:
            if get_distance(self.client_obj, self.obstacles, self.robotID_1[0])<0:
                return False
            if get_distance(self.client_obj, self.obstacles, self.robotID_2[0])<0:
                return False
        return True


def get_numpy_state(state):
    '''
    Returns the numpy state for the dual arm
    :param state: An ompl.base.State object.
    :return np.array: A 14 dimensional numpy array
    '''
    return np.array([state[i] for i in range(14)])

def get_path(start_state, goal_state, si, total_time=10):
    '''
    Planning path from start to goal.
    :param start_state: the start state of the robot.
    :param goal_state: the goal state of the robot.
    :param si: ompl.base.SpaceInformation object.
    :returns tuple: A tuple of planned path, interpolated path, and success flag
    '''
    success = False
    # Define the planning problem
    ss = og.SimpleSetup(si)
    ss.setStartAndGoalStates(start_state, goal_state)

    # Set up planner
    planner = og.RRTstar(si)
    planner.setRange(0.1)
    ss.setPlanner(planner)

    current_time = 45
    solved = ss.solve(current_time)
    while not ss.haveExactSolutionPath() and current_time<total_time:
        solved = ss.solve(1)
        current_time += 1
        
    if ss.haveExactSolutionPath():
        # Save the trajectory
        print("Found solution")
        success = True
        ss.simplifySolution()
        path = [get_numpy_state(ss.getSolutionPath().getState(i))
            for i in range(ss.getSolutionPath().getStateCount())
        ]
        ss.getSolutionPath().interpolate()
        path_interpolated = [get_numpy_state(ss.getSolutionPath().getState(i))
            for i in range(ss.getSolutionPath().getStateCount())
        ]
        return path, path_interpolated, success
    
    return [], [], success

# To collect trajectories for checking self-collision, no obstacles.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', help='Folder where data is saved')
    parser.add_argument('--start', help="start of the samples", type=int)
    parser.add_argument('--samples', help='number of samples to collect', type=int)
    args = parser.parse_args()

    p = get_pybullet_server('gui')
    robot1_ids, robot2_ids = set_dual_robot(p)
    all_obstacles = set_obstacles(p, 0, 8, 8, robot1_ids[0], robot2_ids[0])

    # Get the file directory.
    fileDir = args.log_dir

    space = ob.RealVectorStateSpace(14)
    bounds = ob.RealVectorBounds(14)
    
    # Set joint limits
    for i in range(7):
        bounds.setHigh(i, q_max[0, i])
        bounds.setHigh(i+7, q_max[0, i])
        bounds.setLow(i, q_min[0, i])
        bounds.setLow(i+7, q_min[0, i])
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)
    valid_checker_obj = ValidityCheckerDualDistance(
        p,
        si, 
        robotID_1=(robot1_ids[0], robot1_ids[1]),
        robotID_2=(robot2_ids[0], robot2_ids[1])
    )
    # Run a simple planner:
    si.setStateValidityChecker(valid_checker_obj)

    start_state = ob.State(space)
    goal_state = ob.State(space)

    for i in range(args.start, args.start+args.samples):

        start_state.random()
        while not valid_checker_obj.isValid(start_state()):
            start_state.random()

        goal_state.random()
        while not valid_checker_obj.isValid(goal_state()):
            goal_state.random()


        # Save data
        path, path_interpolated, success = get_path(start_state, goal_state, si)
        if success:
            path_param = {}
            path_param['path'] = path
            path_param['path_interpolated'] = path_interpolated
            path_param['success'] = success

            pickle.dump(path_param, open(osp.join(fileDir,f'path_{i}.p'), 'wb'))
