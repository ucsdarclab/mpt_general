''' Scripts to save constraint planning for the panda arm, on the shelf environment
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


def try_start_location(robotID, jointsID, obstacles):
    '''
    Attempts to place robot at random goal location.
    :param robotID: pybullet ID of the robot.
    :param jointsID: pybullet ID of all obstacles.
    :param obstacles: pybullet ID of all obstacles.
    :returns bool: True if successful.
    '''
    random_pose = (q_min + (q_max-q_min)*np.random.rand(7))[0]
    random_pose[6] = 1.9891
    set_position(robotID, jointsID, random_pose)
    if check_self_collision(robotID) or get_distance(obstacles, robotID)<=0:
        return False
    return True


def get_path(start_state, goal_state, space, all_obstacles, pandaID, jointsID):
    '''
    Use a planner to generate a trajectory from start to goal.
    :param start_state:
    :param goal_state:
    :param space:
    '''
    si = ob.SpaceInformation(space)
    validity_checker_obj = ValidityCheckerDistance(si, all_obstacles, pandaID, jointsID)
    si.setStateValidityChecker(validity_checker_obj)

    # Define planning problem
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_state, goal_state)

    # planner = og.RRT(si)
    planner = og.RRTConnect(si)
    planner.setProblemDefinition(pdef)
    planner.setup()

    # planner.setRange(1)
    dt = 0.25
    totalTime = dt
    solved = planner.solve(dt)
    while not pdef.hasExactSolution():
        solved = planner.solve(dt)
        totalTime += dt
        if totalTime>30:
            break
    plannerData = ob.PlannerData(si)
    planner.getPlannerData(plannerData)
    numVertices = plannerData.numVertices()

    if pdef.hasExactSolution():
        path_simplifier = og.PathSimplifier(si)
        try:
            path_simplifier.simplify(pdef.getSolutionPath(), 0.0)
        except TypeError:
            print("Path not able to simplify for unknown reason!")
            pass
      
        pdef.getSolutionPath().interpolate()
        jointPath = pdef.getSolutionPath().printAsMatrix()
        # Convert path to a numpy array
        jointPath = np.array(list(map(lambda x: np.fromstring(x, dtype=np.float, sep=' '), jointPath.split('\n')))[:-2])
        trajData = {
            'jointPath': jointPath,
            'totalTime': totalTime,
            'numVertices': numVertices
        }
        return trajData, True
    return {'numVertices':numVertices, 'totalTime':totalTime}, False


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
        bounds.setHigh(j, q_max[0, j])
        bounds.setLow(j, q_min[0, j])
    space.setBounds(bounds)
    while i<start+samples:
        # If point is still in collision sample new random points
        valid_goal, goal_joints = try_target_location(client_id, pandaID, jointsID, all_obstacles)
        while not valid_goal:
            valid_goal, goal_joints = try_target_location(client_id, pandaID, jointsID, all_obstacles)

        # get start location
        valid_start = try_start_location(pandaID, jointsID, all_obstacles)
        while not valid_start:
            valid_start = try_start_location(pandaID, jointsID, all_obstacles)
        start_joints = get_joint_position(pandaID, jointsID)

        start_state = get_ompl_state(space, start_joints)
        goal_state = get_ompl_state(space, goal_joints)

        trajData, success = get_path(start_state, goal_state, space,all_obstacles, pandaID, jointsID)
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


def place_shelf(client_obj, base_pose, base_orient):
    '''
    Place the shelf and obstacles in the given pybullet scene.
    :param client_obj: a pybullet scene handle.
    :param base_pose: base position of the shelf.
    :param base_orient: base orientation of the shelf (yaw, pitch, roll) in radians.
    :returns list: obstacles ids of shelf and objects in the shelf.
    '''
    # =================== Place cupboard =================================
    # Where to place the desk
    visualShapeId = client_obj.createVisualShape(shapeType=client_obj.GEOM_MESH,
                                    fileName="assets/cupboard.obj",
                                    rgbaColor=[0.54, 0.31, 0.21, 1],
                                    specularColor=[0.4, .4, 0],
                                    )
    collisionShapeId = client_obj.createCollisionShape(shapeType=client_obj.GEOM_MESH,
                                        fileName="assets/cupboard_vhacd.obj",
                                        )
    base_orientation = pyb.getQuaternionFromEuler(base_orient)
    obstacle_cupboard = client_obj.createMultiBody(baseMass=0.0,
                      baseCollisionShapeIndex=collisionShapeId,
                      baseVisualShapeIndex=visualShapeId,
                      basePosition=base_pose,
                      baseOrientation=base_orientation,
                      useMaximalCoordinates=True
                    )
    return [obstacle_cupboard]


def place_shelf_and_obstacles(client_obj, seed, bp_shelf=[0.3, -0.6, 0.0], bo_shelf=[np.pi/2, 0.0, 0.0], bp_obs=None):
    '''
    Place shelf and obstacles in the environment.
    :param client_obj: a pybullet scene handle object.
    :param seed: The seed used to generate obstacles on the shelf.
    :param bp_shelf: base position of the shelf.
    :param bo_shelf: base orientation of the shelf.
    :param bp_obstacles: base position of the obstacles.
    :return list: all the obstacle ids.
    '''
    if bp_obs is None:
        bp_obs = bp_shelf
    obstacle_cupboard = place_shelf(client_obj, bp_shelf, bo_shelf)

    # Place objects in space.
    shelf_obstacles = set_shelf_obstacles(client_obj, seed, bp_obs)

    return obstacle_cupboard+shelf_obstacles

def generateEnv(start, samples, numPaths, fileDir):
    '''
    Generate environments with randomly placed obstacles
    :param client_id: pybullet client id
    :param start: start index of the environment
    :param samples: Number of samples to collect
    :param numPaths: Number of paths to collect for each environment
    :param fileDir: Directory where path are stored
    '''
    p = get_pybullet_server('direct')
    for seed in range(start, start+samples):
        envFileDir = osp.join(fileDir, f'env_{seed:06d}')
        if not osp.isdir(envFileDir):
            os.mkdir(envFileDir)

        # Reset the environment
        set_simulation_env(p)
        # Place obstacles in space.
        all_obstacles = place_shelf_and_obstacles(p, seed)
        # Spawn the robot.
        pandaID, jointsID, fingersID = set_robot(p)

        start_experiment_rrt(0, numPaths, envFileDir, pandaID, jointsID, all_obstacles)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planning for RRT for Panda robot")
    parser.add_argument('--start', help="start of the sample index", required=True, type=int)
    parser.add_argument('--samples', help="Number of samples to collect", required=True, type=int)
    parser.add_argument('--fileDir', help="Folder to save the files", required=True)
    parser.add_argument('--numPaths', type=int)
    args = parser.parse_args()

    generateEnv(args.start, args.samples, args.numPaths, args.fileDir)