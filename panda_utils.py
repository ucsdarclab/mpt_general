''' Contains useful functions to interface with the panda arm.
'''

import pybullet as pyb
import pybullet_data
import pybullet_utils.bullet_client as bc

import numpy as np
import pickle
import os

# Set up path planning
from ompl import base as ob
from ompl import geometric as og

import warnings

# Panda limits
q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698,
                 2.8973, 3.7525, 2.8973])[None, :]
q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -
                 2.8973, -0.0175, -2.8973])[None, :]


def get_numpy_state(state):
    ''' Return the state as a numpy array.
    :param state: An ob.State from ob.RealVectorStateSpace
    :return np.array:
    '''
    return np.array([state[i] for i in range(7)])


# Spawn robot
def set_robot(client_obj, base_pose = np.array([0]*3), base_orientation=np.array([0.0]*3)):
    ''' Spawn the robot in the environment.
    :param client_obj: a pybullet_utils.BulletClient object
    :param base_pose: the base position of the panda arm.
    :param base_orientation: the base orientation of the panda arm in (yaw, pitch, roll)
    '''
    base_ori_quat = pyb.getQuaternionFromEuler(base_orientation)
    panda = client_obj.loadURDF(
        "franka_panda/panda.urdf", 
        basePosition=base_pose, 
        baseOrientation=base_ori_quat,
        flags=pyb.URDF_USE_SELF_COLLISION,
        useFixedBase=True
    )
    # Get the joint info
    numLinkJoints = client_obj.getNumJoints(panda)
    jointInfo = [client_obj.getJointInfo(panda, i) for i in range(numLinkJoints)]
    # Joint nums
    joints = [j[0] for j in jointInfo if j[2]==pyb.JOINT_REVOLUTE]
    # finger nums
    finger = [j[0] for j in jointInfo if j[2]==pyb.JOINT_PRISMATIC]
    return panda, joints, finger

# Spawn robot for visualization purposes only.
def set_robot_vis(client_obj, rgbaColor=None, base_pose = np.array([0]*3), base_orientation=np.array([0.0]*3)):
    '''
    Spawn a new robot at the proposed pose, and set its transparency to the set value.
    :param client_obj: A pybullet_utils.BulletClient object.
    :param pose: The pose of the robot.
    :param rgbaColor: Color of the robot.
    '''
    # Spawn the robot.
    base_ori_quat = pyb.getQuaternionFromEuler(base_orientation)
    pandaVis = client_obj.loadURDF(
        "franka_panda/panda.urdf", 
        basePosition=base_pose,
        baseOrientation=base_ori_quat,
        flags=pyb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    )
    # Get the joint info
    numLinkJoints = client_obj.getNumJoints(pandaVis)
    jointInfo = [client_obj.getJointInfo(pandaVis, i) for i in range(numLinkJoints)]
    # Joint nums
    jointsVis = [j[0] for j in jointInfo if j[2]==pyb.JOINT_REVOLUTE]
    # Set the robot to a particular pose.
    set_position(client_obj, pandaVis, jointsVis, base_pose)
    # Change the color of the robot.
    for j in range(numLinkJoints):
        client_obj.changeVisualShape(pandaVis, j, rgbaColor=rgbaColor)
    return pandaVis, jointsVis, []

    
def set_simulation_env(client_obj):
    '''
    Set environment for the given client object.
    :param client_obj: A pybullet_utils.BulletClient object.
    '''
    client_obj.resetSimulation()
    client_obj.setGravity(0, 0, -9.8)

def set_position(client_obj, model, joints, jointValue):
    ''' Set the model robot to the given joint values
    :param model: pybullet id of link.
    :param jointValue: joint value to be set
    '''
    for jV, j in zip(jointValue, joints):
        client_obj.resetJointState(model, j, jV)

def get_random_pos(num_points):
    ''' Generate random points in 3D space.
    :param num_points: Number of points to be generated.
    :returns np.array: Returns an array of (num_points, 3)
    '''
    R = np.random.rand(num_points)*0.8 + 0.4
    Phi = np.random.rand(num_points)*3*np.pi/4 - np.pi/4
    Theta = np.random.rand(num_points)*np.pi*2
    XYZ = np.c_[(R*np.cos(Phi)*np.cos(Theta), R*np.cos(Phi)*np.sin(Theta), R*np.sin(Phi))]
    return XYZ

# Spawn obstacles
box_length = box_width = 0.2
sph_radius = 0.15
rgba = [0.125, 0.5, 0.5, 1]

def set_obstacles(client_obj, seed, num_boxes, num_spheres):
    '''Set box and spheres in the environment
    :param client_obj: A pybullet_utils.BulletClient object
    :param seed: random seed to be set.
    :param num_boxes: Number of boxes to be used.
    :param num_sphere: Number of spheres to be used.
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
    # Define square objects, position in spherical co-ordinates
    boxXYZ = get_random_pos(num_points=num_boxes)
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

    # Define spherical objects, position in spherical co-ordinates
    sphXYZ = get_random_pos(num_points=num_spheres)
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
    return obstacles_box+obstacles_sph

# Collision checking
def get_distance(client_obj, obstacles, robotID):
    '''  
    Return the distance of the obstacles and robot.
    :param obstacles: A list of obstacle ID
    :param robotID: ID of the robot
    :returns float: The minimum distance of the robot and obstacle
    '''
    assert isinstance(obstacles, list), "Obstacles has to be a list"
    distance =  min(
            (
                min(link[8] for link in client_obj.getClosestPoints(bodyA=obs, bodyB=robotID, distance=100))
                for obs in obstacles
            )
        )
    return distance


# Create offsets for checking link collision.
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
link_offset = np.diag(selfContact)+np.diag(adjContact, k=1)+ np.diag(adjContact, k=-1)
def check_self_collision(client_id, robotID):
    ''' Checks if the robot meshes are touching each other.
    :param robotID: the pybullet ID of the robot.
    :returns bool: Returns True if the robot links are in collision
    '''
    collision_mat = np.array([link[8] for link in client_id.getClosestPoints(robotID, robotID, distance=2)]).reshape((11, 11))
    collMat = collision_mat-link_offset
    minDist = np.min(collMat)
    return minDist<0 and not np.isclose(minDist, 0.0)


class ValidityCheckerDistance(ob.StateValidityChecker):
    '''A class to check the validity of the state, by checking distance function
    '''
    defaultOrientation = pyb.getQuaternionFromEuler([0., 0., 0.])
    def __init__(self, client_obj, si, obstacles, robotID, joints):
        '''
        Initialize the class object, with the obstacle ID's
        :param si: an object of type omp.base.SpaceInformation
        :param obstacles: A list of obstacle ID
        :param robotID: ID of the robot
        :param joints: list of joints
        '''
        super().__init__(si)
        self.client_obj = client_obj
        self.obstacles = obstacles
        self.robotID = robotID
        self.joints = joints
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
        self.offset = np.diag(selfContact)+np.diag(adjContact, k=1)+ np.diag(adjContact, k=-1)

    def isValid(self, state):
        '''
        Check if the given state is valid.
        :param state: An ob.STate object to be checked.
        :return bool: True if the state is valid.
        '''
        # Set robot position
        set_position(self.client_obj, self.robotID, self.joints, [state[i] for i in range(7)])
        if not check_self_collision(self.client_obj, self.robotID):
            if self.obstacles is not None:
                return self.getDistance(state)>0
            else:
                return True
        return False

    def getDistance(self, state):
        '''
        Get the shortest distance from robot to obstacle.
        :param x: A numpy array of state x.
        :returns float: The closest distance between the robot and obstacle.
        '''
        return get_distance(self.client_obj, self.obstacles, self.robotID)

    def getCollisionMat(self):
        '''Get collision matrix between links
        '''
        warnings.warn("Use the function, not the class objects")
        return np.array([link[8] for link in self.client_obj.getClosestPoints(self.robotID, self.robotID, distance=2)]).reshape((11, 11))

    def checkSelfCollision(self):
        '''Returns True if links are in self-collision for the PANDA robot
        '''
        warnings.warn("Use the function, not the class objects")
        collMat = self.getCollisionMat()-self.offset
        minDist = np.min(collMat)
        return minDist<0 and not np.isclose(minDist, 0.0)

def get_pybullet_server(connection_type):
    '''
    Returns the pybullet object, after creating the server.
    :param connection_type: GUI/DIRECT
    :returns bc.BulletClient:
    '''
    if connection_type=='direct':
        p = bc.BulletClient(pyb.DIRECT)
    elif connection_type=='gui':
        p = bc.BulletClient(pyb.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    else:
        raise TypeError
    p.setGravity(0, 0, -9.81)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    return p

def set_env(client_obj, space, num_boxes, num_spheres, seed):
    '''
    Generate environment with randomly placed obstacles in space.
    :param client_obj: bc.BulletClient object
    :param space: An ompl space object.
    :param num_boxes:
    :param num_spheres:
    :param seed:
    :returns ValidityCheckerObj:
    '''

    set_simulation_env(client_obj)
    panda, joints, _ = set_robot(client_obj)
    obstacles = set_obstacles(client_obj, num_boxes=num_boxes, num_spheres=num_spheres, seed = seed)
    si = ob.SpaceInformation(space)   
    ValidityCheckerObj = ValidityCheckerDistance(client_obj, si, obstacles, panda, joints)
    return ValidityCheckerObj


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

    current_time = 2
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