''' Create a panda environment for shelf plannng
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

from panda_utils import q_max, q_min, get_pybullet_server, set_simulation_env
from panda_utils import set_robot, set_position, get_distance
from panda_utils import ValidityCheckerDistance
from ompl_utils import get_ompl_state ,get_numpy_state

def set_IK_position(client_obj, model, joints, end_effector_pose, end_effector_orient=None):
    '''
    Calcualtes the IK and places the robot hand at the given end-effector position.
    :param model: pybullet id of robot.
    :param joints: pybullet id of robot link.
    :param end_effector_pose: 3D pose of the end-effector
    :param end_effector_orient: 3D orientation of the end-effector in euler angle.
    '''
    if end_effector_orient is not None:
        end_effector_orient_quat = pyb.getQuaternionFromEuler(end_effector_orient)
        joint_pose = client_obj.calculateInverseKinematics(
            model, 
            8, 
            end_effector_pose, 
            end_effector_orient_quat,
            q_min[0],
            q_max[0],
            (q_max-q_min)[0],
            (q_max+q_min)[0]/2,
            maxNumIterations=75
        )
    else:
        joint_pose = client_obj.calculateInverseKinematics(
            model, 
            8, 
            end_effector_pose, 
            lowerLimits=q_min[0],
            upperLimits=q_max[0],
            jointRanges=(q_max-q_min)[0],
            restPoses=(q_max+q_min)[0]/2,
            maxNumIterations=75
        )
    set_position(client_obj, model, joints, joint_pose)
    return joint_pose


def get_robot_end_effector_pose(client_id, robotID):
    '''
    Returns the cartesian pose of the end-effector
    '''
    return np.array(client_id.getLinkState(robotID, 8)[4])


def check_self_collision(client_id, robotID):
    '''
    Checks if the robot is in collision with itself.
    :param robotID: pybullet ID of the robot.
    :returns bool: True if links are in self-collision for the PANDA robot.
    '''
    # Create offsets for the meshes.
    selfContact = np.diag([
        -0.1420203 , -0.11206833, -0.11204374, -0.12729175, -0.12736233,
        -0.1114051 , -0.09016624, -0.05683277, -0.06448606, -0.02296515,
        -0.02296515
        ])
    adjContact = np.array([
        -0.0009933453129464674, -0.03365764455472735, -0.0010162024664227207, 
        -0.04104534748867784, -0.006549498247919756, -0.05069805702950261, 
        -0.0012093463397529107, -0.02519203810879126, -0.009414129012614625,
        -0.002214306228527925
        ])
    offset = np.diag(selfContact)+np.diag(adjContact, k=1)+ np.diag(adjContact, k=-1)
    
    collMat = np.array(
        [link[8] for link in client_id.getClosestPoints(robotID, robotID, distance=2)]
    ).reshape((11, 11))-offset
    minDist = np.min(collMat)
    return minDist<0 and not np.isclose(minDist, 0.0)

def set_shelf_obstacles(client_obj, seed, base_shelf_position):
    '''
    Set the obstacles on the shelf.
    :param client_obj: Bullet client
    :param seed: seed used to generate obstacles in the shelf.
    :returns list: ID of shelf obstacles.
    '''
    # Define the cylinderical obstacle
    sph_radius = 0.05
    rgba = [0.125, 0.5, 0.5, 1]
    geomCylinder = client_obj.createCollisionShape(pyb.GEOM_CYLINDER, radius=sph_radius, height= 0.2)
    visualCylinder = client_obj.createVisualShape(pyb.GEOM_CYLINDER, radius=sph_radius, length=0.2, rgbaColor=rgba)
    shelf_obstacles = []
    scale = np.array([0.4, 0.0, 0.0])
    np.random.seed(seed)
    select_shelf = np.random.random_integers(0, 1, 3)
    shelf_1_offset = np.array([0.1, 0.2, 0.1])
    shelf_2_offset = np.array([0.1, 0.2, 0.38])
    shelf_3_offset = np.array([0.1, 0.2, 0.64])
    if select_shelf[0]==1:
        # Shelf 1
        shelf_obstacles.append(
            client_obj.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=geomCylinder,
                baseVisualShapeIndex=visualCylinder,
                basePosition=base_shelf_position+shelf_1_offset+np.random.rand()*scale,
            )
        )
    if select_shelf[1]==1:
        # Shelf 2
        shelf_obstacles.append(
            client_obj.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=geomCylinder,
                baseVisualShapeIndex=visualCylinder,
                basePosition=base_shelf_position+shelf_2_offset+np.random.rand()*scale,
            )
        )
    if select_shelf[2]==1:
        # Shelf 3
        shelf_obstacles.append(
            client_obj.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=geomCylinder,
                baseVisualShapeIndex=visualCylinder,
                basePosition=base_shelf_position+shelf_3_offset+np.random.rand()*scale,
            )
        )
    return shelf_obstacles


def get_joint_position(client_id, robotID, jointsID):
    '''
    Returns a numpy array of all the joints for the given robot.
    :param robotID: pybullet ID of the robot.
    :param jointsID: List of link ID whose joint angles are requested.
    :return np.array: The resulting joint configuration.
    '''
    return np.array(list(map(lambda x:x[0], client_id.getJointStates(robotID, jointsID))))

def check_bounds(joint_angle):
    '''
    Returns true if the given joint angles are within bounds.
    :param joint_angle: np.array of 7 joints
    :returns bool: If true the the joints are within bounds
    '''
    return (joint_angle<=q_max[0]).all() and (joint_angle>=q_min[0]).all()

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
    set_joint_pose = np.array(set_IK_position(client_obj, robotID, jointsID, random_pose, random_orient))[:7]
    # Check if the robot is in self-collision/collision w/ obstacles.
    if check_self_collision(client_obj, robotID) or get_distance(client_obj, obstacles, robotID)<=0 or (not check_bounds(set_joint_pose)):
        # If so randomize the joints and calculate IK once again.
        random_joint_pose = (q_min + (q_max - q_min)*np.random.rand(7))[0]
        set_position(client_obj, robotID, jointsID, random_joint_pose)
        set_joint_pose = np.array(set_IK_position(client_obj, robotID, jointsID, random_pose, random_orient))[:7]
        if check_self_collision(client_obj, robotID) or get_distance(client_obj, obstacles, robotID)<=0 or (not check_bounds(set_joint_pose)):
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
    random_pose = (q_min + (q_max-q_min)*np.random.rand(7))[0]
    random_pose[6] = 1.9891
    set_position(client_obj, robotID, jointsID, random_pose)
    if check_self_collision(client_obj, robotID) or get_distance(client_obj, obstacles, robotID)<=0:
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
        valid_start = try_start_location(client_id, pandaID, jointsID, all_obstacles)
        while not valid_start:
            valid_start = try_start_location(client_id, pandaID, jointsID, all_obstacles)
        start_joints = get_joint_position(client_id, pandaID, jointsID)

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