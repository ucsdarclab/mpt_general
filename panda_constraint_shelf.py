''' Scripts to save constraint planning for the panda arm, on the shelf environment
'''

import pybullet as pyb
import pybullet_data
import pybullet_utils.bullet_client as bc

import roboticstoolbox as rtb

import numpy as np
import pickle
import os
from os import path as osp

# Set up path planning
from ompl import base as ob
from ompl import geometric as og

import argparse
import time

import panda_utils as pu
import ompl_utils as ou
import panda_shelf_env as pse

def try_target_location(client_obj, robotID, jointsID,  obstacles):
    '''
    Attempts to place robot at random target location.
    :param robotID: pybullet ID of the robot.
    :param obstacles: pybullet ID of all obstacles.
    :returns bool: True if succcessful.
    '''
    base_pose = np.array([0.45, -0.44, 0.1])
    scale = np.array([0.45, 0.16, 0.6])
    # TODO: Randomize target orientation.
    random_orient = [np.pi/2, -np.pi/2, 0.0]
    random_pose = base_pose + scale*np.random.rand(3)
    set_joint_pose = np.array(pse.set_IK_position(client_obj, robotID, jointsID, random_pose, random_orient))[:7]
    # Check if the robot is in self-collision/collision w/ obstacles.
    if pu.check_self_collision(robotID) or pu.get_distance(obstacles, robotID)<=0 or (not pse.check_bounds(set_joint_pose)):
        # If so randomize the joints and calculate IK once again.
        random_joint_pose = (pu.q_min + (pu.q_max - pu.q_min)*np.random.rand(7))[0]
        pu.set_position(robotID, jointsID, random_joint_pose)
        set_joint_pose = np.array(pse.set_IK_position(client_obj, robotID, jointsID, random_pose, random_orient))[:7]
        if pu.check_self_collision(robotID) or pu.get_distance(obstacles, robotID)<=0 or (not pse.check_bounds(set_joint_pose)):
            return False, set_joint_pose
    return True, set_joint_pose


def try_start_location(client_obj, robotID, jointsID, obstacles):
    '''
    Attempts to place robot at random goal location.
    :param robotID: pybullet ID of the robot.
    :param jointsID: pybullet ID of all obstacles.
    :param obstacles: pybullet ID of all obstacles.
    :returns bool: True if successful.
    '''
    random_pose = (pu.q_min + (pu.q_max-pu.q_min)*np.random.rand(7))[0]
    pu.set_position(robotID, jointsID, random_pose)
    link_state = pyb.getLinkState(robotID, 8)
    random_orient = [np.pi/2, -np.pi/2, 0.0]
    joint_pose = np.array(pse.set_IK_position(client_obj, robotID, jointsID, link_state[0], random_orient))[:7]
    if pu.check_self_collision(robotID) or pu.get_distance(obstacles, robotID)<=0 or (not pse.check_bounds(joint_pose)):
        return False
    return True


# TODO: Write the planner for the constraint function
def get_constraint_path(start_config, goal_config, goal_ori, space, all_obstacles, pandaID, jointsID):
    '''
    Use a planner to generate a trajectory from start to goal.
    :param start_state:
    :param goal_state:
    :param space:
    '''
    tolerance = np.array([0.1, 2*np.pi, 2*np.pi])
    constraint_function = EndEffectorConstraint(goal_ori, tolerance, pandaID, jointsID)

    # Set up the constraint planning space.
    css = ob.ProjectedStateSpace(space, constraint_function)
    csi = ob.ConstrainedSpaceInformation(css)

    ss = og.SimpleSetup(csi)
    validity_checker = pu.ValidityCheckerDistance(csi, all_obstacles, pandaID, jointsID)
    ss.setStateValidityChecker(validity_checker)

    # Define the start and goal state
    start_state = ob.State(csi.getStateSpace())
    goal_state = ob.State(csi.getStateSpace())

    for i in range(7):
        start_state[i] = start_config[i]
        goal_state[i] = goal_config[i]

    # Define planning problem
    ss.setStartAndGoalStates(start_state, goal_state)

    # planner = og.RRT(si)
    planner = og.RRTConnect(csi)
    ss.setPlanner(planner)
    ss.setup()

    # planner.setRange(1)
    total_time = 0
    success = False
    while not success and total_time<300:
        solved = ss.solve(30)
        success = ss.haveExactSolutionPath()
        total_time += 30
        
    plannerData = ob.PlannerData(csi)
    planner.getPlannerData(plannerData)
    numVertices = plannerData.numVertices()

    if success:
        ss.simplifySolution()
        path = [get_numpy_state(ss.getSolutionPath().getState(i))
            for i in range(ss.getSolutionPath().getStateCount())
        ]
        ss.getSolutionPath().interpolate()
        path_interpolated = ss.getSolutionPath().printAsMatrix()
        # Convert path to a numpy array
        path_interpolated = np.array(list(map(lambda x: np.fromstring(x, dtype=np.float32, sep=' '), path_interpolated.split('\n')))[:-2])
        trajData = {
            'path': np.array(path),
            'path_interpolated': path_interpolated,
            'totalTime': total_time,
            'numVertices': numVertices
        }
        return trajData, True
    return {'numVertices':numVertices, 'totalTime':total_time}, False


def start_experiment_rrt(client_id, start, samples, fileDir, pandaID, jointsID, all_obstacles):
    '''
    Run the experiment for random start and goal points.
    :param client_id: Bullet client
    :param start: the start index of the samples.
    :param samples: the number of samples to collect
    :param fileDir: Directory to store the path details
    :param (pandaID, jointsID, all_obstacles): pybullet ID's of panda arm, joints and obstacles in space
    '''
    assert osp.isdir(fileDir), f"{fileDir} is not a valid directory"
    i = start
    tryCount = 0
    # Planning parameters
    space = ob.RealVectorStateSpace(7)
    bounds = ob.RealVectorBounds(7)
    # Set joint limits
    for j in range(7):
        bounds.setHigh(j, pu.q_max[0, j])
        bounds.setLow(j, pu.q_min[0, j])
    space.setBounds(bounds)
    while i<start+samples:
        # If point is still in collision sample new random points
        valid_goal, goal_joints = try_target_location(client_id, pandaID, jointsID, all_obstacles)
        while not valid_goal:
            valid_goal, goal_joints = try_target_location(client_id, pandaID, jointsID, all_obstacles)
        goal_pose, goal_ori = get_robot_pose_orientation(pandaID)

        # get start location
        valid_start = try_start_location(client_id, pandaID, jointsID, all_obstacles)
        while not valid_start:
            valid_start = try_start_location(client_id, pandaID, jointsID, all_obstacles)
        start_joints = pse.get_joint_position(pandaID, jointsID)

        trajData, success = get_constraint_path(start_joints, goal_joints, goal_ori, space,all_obstacles, pandaID, jointsID)
        tryCount +=1
        # if tryCount>5:
        #     i +=1
        #     tryCount=0

        if success:
            print(f"Found Path {i}")
            trajData['success'] = success
            pickle.dump(trajData, open(osp.join(fileDir, f'path_{i}.p'), 'wb'))
            i += 1
            tryCount = 0

def generateEnv(start, samples, numPaths, fileDir):
    '''
    Generate environments with randomly placed obstacles
    :param client_id: pybullet client id
    :param start: start index of the environment
    :param samples: Number of samples to collect
    :param numPaths: Number of paths to collect for each environment
    :param fileDir: Directory where path are stored
    '''
    p = pu.get_pybullet_server('direct')
    for seed in range(start, start+samples):
        envFileDir = osp.join(fileDir, f'env_{seed:06d}')
        if not osp.isdir(envFileDir):
            os.mkdir(envFileDir)

        # Reset the environment
        pu.set_simulation_env(p)
        # Place obstacles in space.
        all_obstacles = pse.place_shelf_and_obstacles(p, seed)
        # Spawn the robot.
        pandaID, jointsID, fingersID = pu.set_robot(p)

        start_experiment_rrt(p, 0, numPaths, envFileDir, pandaID, jointsID, all_obstacles)

def quaternion_difference(q1, q2):
    """
    NOTE: This function was generated using ChatGPT!!! Use w/ caution.
    Calculates the difference between two orientations represented using quaternions.

    Args:
        q1 (tuple): A tuple representing the first quaternion in the format (x, y, z, w).
        q2 (tuple): A tuple representing the second quaternion in the format (x, y, z, w).

    Returns:
        tuple: A tuple representing the quaternion difference in the format (w_diff, x_diff, y_diff, z_diff).
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    # Calculate the conjugate of q2
    q2_conjugate = (w2, -x2, -y2, -z2)

    # Multiply q1 by the conjugate of q2
    diff = (
        w1 * q2_conjugate[1] + x1 * q2_conjugate[0] + y1 * q2_conjugate[3] - z1 * q2_conjugate[2],
        w1 * q2_conjugate[2] - x1 * q2_conjugate[3] + y1 * q2_conjugate[0] + z1 * q2_conjugate[1],
        w1 * q2_conjugate[3] + x1 * q2_conjugate[2] - y1 * q2_conjugate[1] + z1 * q2_conjugate[0],
        w1 * q2_conjugate[0] - x1 * q2_conjugate[1] - y1 * q2_conjugate[2] - z1 * q2_conjugate[3]
    )

    return diff

def get_robot_pose_orientation(robotID):
    '''
    Returns the world pose and orientation of the robot end-effector.
    :param client_id: pybullet client object.
    :param robotID: pybullet robot id.
    :returns tuple: (Pose, orientation) the pose and orientation of the robot.
    '''
    link_state = pyb.getLinkState(robotID, 11)
    return link_state[4], link_state[5]


# Constraint function
def angularVelociyToAngleAxis(angle, axis):
    '''
    Return a matrix to convert angular velocity to angle-axis velocity
    :apram angle: 
    :param axis: 
    '''
    t = abs(angle)
    R_skew = np.array([
            [ 0,    -axis[2],  axis[1]],
            [ axis[2],  0,    -axis[0]],
            [-axis[1],  axis[0],  0]
        ])
    c = (1-0.5*t*np.sin(t)/(1-np.cos(t)))
    return np.eye(3) - 0.5*R_skew+(R_skew@R_skew/(t*t))*c

class EndEffectorConstraint(ob.Constraint):
    def __init__(self, target_ori, tolerance, robotID, jointsID):
        '''
        Initialize the constraint function for fixed orientation constraint.
        :param target_ori: The target orientation set by the planner represented using [x, y, z, w]
        :param tolerance: wiggle room available for each axis of rotation.
        :param robotID: pybullet ID of robot model
        '''
        self.target_ori = target_ori
        self.tolerance = tolerance
        self.robotID = robotID
        self.jointsID = jointsID
        self.panda_model = rtb.models.DH.Panda()
        super().__init__(7, 3)

    def get_current_position(self, q):
        '''
        Set the robot to the given position and return orientation.
        '''
        pu.set_position(self.robotID, self.jointsID, q)
        return get_robot_pose_orientation(self.robotID)[1]

        
    def function(self, x, out):
        '''
        Evaluate the value of the function at the given point x
        and set its value to out.
        :param x: value at state.
        :param out: constraint value.
        '''
        # Using rtb
        # orient_diff = self.panda_model.fkine(x).R.T@self.fix_orient_R
        orient_diff = self.fix_orient_R.T@self.panda_model.fkine(x).R
        angle, axis = spm.base.tr2angvec(orient_diff)
        axis_error = angle*axis
        for i in range(3):
            if  abs(axis_error[i])<self.tolerance[i]:
                out[i] = 0.0
            else:
                out[i] = abs(axis_error[i])
    
    def bound_derivative(self, error):
        '''
        Returns the jacobian of the bound.
        '''
        gradient = np.zeros((3, 3))
        for i, e_i in enumerate(error):
            if abs(e_i)>self.tolerance[i]:
                gradient[i, i] = e_i/abs(e_i)
        return gradient

    # Jacobian 1
    def jacobian(self, x, out):
        # Using rbt
        orient_diff = self.fix_orient_R.T@self.panda_model.fkine(x).R
        angle, axis = spm.base.tr2angvec(orient_diff)
        bound_gradient = self.bound_derivative(angle*axis)
        error_jacobian = bound_gradient@self.fix_orient_R.T@self.panda_model.jacob0(x, half='rot')
        for r in range(3):
            for c in range(7):
                out[r, c] = error_jacobian[r, c]
        out = error_jacobian.copy()


def get_numpy_state(state):
    ''' Return the state as a numpy array.
    :param state: An ob.State from ob.RealVectorStateSpace
    :return np.array:
    '''
    return np.array([state[i] for i in range(7)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planning for RRT for Panda robot")
    parser.add_argument('--start', help="start of the sample index", required=True, type=int)
    parser.add_argument('--samples', help="Number of samples to collect", required=True, type=int)
    parser.add_argument('--fileDir', help="Folder to save the files", required=True)
    parser.add_argument('--numPaths', type=int)
    args = parser.parse_args()

    generateEnv(args.start, args.samples, args.numPaths, args.fileDir)