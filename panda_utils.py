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

# Panda limits
q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698,
                 2.8973, 3.7525, 2.8973])[None, :]
q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -
                 2.8973, -0.0175, -2.8973])[None, :]


connection_mode = pyb.DIRECT
p = bc.BulletClient(connection_mode=connection_mode)
pyb.setAdditionalSearchPath(pybullet_data.getDataPath())

# Spawn robot
def set_robot(client_obj):
    ''' Spawn the robot in the environment.
    :param client_obj: a pybullet_utils.BulletClient object
    '''
    panda = client_obj.loadURDF("franka_panda/panda.urdf", np.array([0, 0, 0]), flags=pyb.URDF_USE_SELF_COLLISION)
    # Get the joint info
    numLinkJoints = pyb.getNumJoints(panda)
    jointInfo = [pyb.getJointInfo(panda, i) for i in range(numLinkJoints)]
    # Joint nums
    joints = [j[0] for j in jointInfo if j[2]==pyb.JOINT_REVOLUTE]
    # finger nums
    finger = [j[0] for j in jointInfo if j[2]==pyb.JOINT_PRISMATIC]
    return panda, joints, finger

def set_simulation_env(client_obj):
    '''
    Set environment for the given client object.
    :param client_obj: A pybullet_utils.BulletClient object.
    '''
    client_obj.resetSimulation()
    client_obj.setGravity(0, 0, -9.8)

def set_position(model, joints, jointValue):
    ''' Set the model robot to the given joint values
    :param model: pybullet id of link.
    :param jointValue: joint value to be set
    '''
    for jV, j in zip(jointValue, joints):
        pyb.resetJointState(model, j, jV)

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
def get_distance(obstacles, robotID):
    '''  
    Return the distance of the obstacles and robot.
    :param obstacles: A list of obstacle ID
    :param robotID: ID of the robot
    :returns float: The minimum distance of the robot and obstacle
    '''
    assert isinstance(obstacles, list), "Obstacles has to be a list"
    distance =  min(
            (
                min(link[8] for link in pyb.getClosestPoints(bodyA=obs, bodyB=robotID, distance=100))
                for obs in obstacles
            )
        )
    return distance

class ValidityCheckerDistance(ob.StateValidityChecker):
    '''A class to check the validity of the state, by checking distance function
    '''
    defaultOrientation = pyb.getQuaternionFromEuler([0., 0., 0.])
    def __init__(self, si, obstacles, robotID, joints):
        '''
        Initialize the class object, with the obstacle ID's
        :param si: an object of type omp.base.SpaceInformation
        :param obstacles: A list of obstacle ID
        :param robotID: ID of the robot
        :param joints: list of joints
        '''
        super().__init__(si)
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
        set_position(self.robotID, self.joints, [state[i] for i in range(7)])
        if not self.checkSelfCollision():
            return self.getDistance(state)>0
        return False

    def getDistance(self, state):
        '''
        Get the shortest distance from robot to obstacle.
        :param x: A numpy array of state x.
        :returns float: The closest distance between the robot and obstacle.
        '''
        return get_distance(self.obstacles, self.robotID)

    def getCollisionMat(self):
        '''Get collision matrix between links
        '''
        return np.array([link[8] for link in pyb.getClosestPoints(self.robotID, self.robotID, distance=2)]).reshape((11, 11))

    def checkSelfCollision(self):
        '''Returns True if links are in self-collision for the PANDA robot
        '''
        collMat = self.getCollisionMat()-self.offset
        minDist = np.min(collMat)
        return minDist<0 and not np.isclose(minDist, 0.0)

def set_env(space, num_boxes, num_spheres, seed):
    '''
    Generate environment with randomly placed obstacles in space.
    :param space: An ompl space object.
    :param num_boxes:
    :param num_spheres:
    :param seed:
    :returns ValidityCheckerObj:
    '''
    set_simulation_env(p)
    panda, joints, _ = set_robot(p)
    obstacles = set_obstacles(p, num_boxes=num_boxes, num_spheres=num_spheres, seed = seed)
    si = ob.SpaceInformation(space)   
    ValidityCheckerObj = ValidityCheckerDistance(si, obstacles, panda, joints)
    return ValidityCheckerObj