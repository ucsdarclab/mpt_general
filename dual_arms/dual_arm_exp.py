''' Dual arm experiment setup for passing object task.
'''

import os
import sys
# To make sure this can access all packages in the previous folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pybullet as pyb
import numpy as np

import panda_utils as pdu
import dual_arm_utils as dau
import collect_data as cd
import panda_shelf_env as pse
from panda_utils import box_length, box_width, rgba, sph_radius


def set_IK_position(client_obj, model, joints, end_effector_id, end_effector_pose, end_effector_orient=None):
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
            end_effector_id, 
            end_effector_pose, 
            end_effector_orient_quat,
            pdu.q_min[0],
            pdu.q_max[0],
            (pdu.q_max-pdu.q_min)[0],
            (pdu.q_max+pdu.q_min)[0]/2,
            maxNumIterations=75
        )
    else:
        joint_pose = client_obj.calculateInverseKinematics(
            model, 
            end_effector_id, 
            end_effector_pose, 
            lowerLimits=pdu.q_min[0],
            upperLimits=pdu.q_max[0],
            jointRanges=(pdu.q_max-pdu.q_min)[0],
            restPoses=(pdu.q_max+pdu.q_min)[0]/2,
            maxNumIterations=75
        )
    pdu.set_position(client_obj, model, joints, joint_pose)
    return joint_pose

def get_robot_pose(client_id, robotID, ee_link_id):
    '''
    Returns the cartesian pose of the end-effector
    '''
    return np.array(client_id.getLinkState(robotID, ee_link_id)[4])

def set_obstacles(client_obj, seed, num_boxes, num_spheres, robot_id1, robot_id2):
    ''' Place obstacles around the current robot position.
    :param client_obj: A pybullet_utils.BulletClient object.
    :param seed: Random seed to be set.
    :param num_boxes: Number of boxes to be used.
    :param num_spheres: Number of spheres to be used.
    :return list: returns the ids of obstacles set in the simulation.
    '''
    # Define the box obstacle
    geomBox = client_obj.createCollisionShape(pyb.GEOM_BOX, halfExtents=[box_length/2, box_width/2, box_width/2])
    visualBox = client_obj.createVisualShape(pyb.GEOM_BOX, halfExtents=[box_length/2, box_width/2, box_width/2], rgbaColor=rgba)
    # Define the spherical obstacle
    geomSphere = client_obj.createCollisionShape(pyb.GEOM_SPHERE, radius=sph_radius)
    visualSphere = client_obj.createVisualShape(pyb.GEOM_SPHERE, radius=sph_radius, rgbaColor=rgba)

    np.random.seed(seed)
    robot_base_pose = [np.r_[0.42, 0., 0.], np.r_[-0.42, 0., 0.]]

    # Define square objects, position in spherical co-ordinates
    boxXYZ = pdu.get_random_pos(num_points=num_boxes)
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
    # Check if the obstacles are in collision with the either robot
    new_obstacles_box = [obs 
        for obs in obstacles_box 
            if not (pdu.get_distance(client_obj, [obs], robot_id1)<0 or pdu.get_distance(client_obj, [obs], robot_id2)<0)
    ]
    # Remove the obstacles from env
    for obs in obstacles_box:
        if obs not in new_obstacles_box:
            pyb.removeBody(obs)

    # Define spherical objects, position in spherical co-ordinates
    sphXYZ = pdu.get_random_pos(num_points=num_spheres)
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
            if not (pdu.get_distance(client_obj, [obs], robot_id1)<0 or pdu.get_distance(client_obj, [obs], robot_id2)<0)
    ]
    for obs in obstacles_sph:
        if obs not in new_obstacles_sph:
            pyb.removeBody(obs)

    return new_obstacles_box+new_obstacles_sph

def get_joint_position(client_obj, robotid, jointids):
    '''
    return a numpy array of joint states.
    :param robotid:
    :param jointids:
    '''
    return np.array([client_obj.getJointState(robotid, ji)[0] for ji in jointids])

if __name__=="__main__":
    p = pdu.get_pybullet_server('gui')
    robotid1, robotid2 = dau.set_dual_robot(p)

    get_random_pose = lambda : ((pdu.q_max-pdu.q_min)*np.random.rand(7) + pdu.q_min).squeeze()
    # Find valid robot 1 goal pose
    # Set robot1 to random pose
    pdu.set_position(p, robotid1[0], robotid1[1], get_random_pose())
    robot1_goal_pose = get_joint_position(p, robotid1[0], robotid1[1])

    random_pose = np.r_[-0.095, 0.5, 0.5]
    random_orient = np.r_[-np.pi, -np.pi/2, np.pi]
    set_IK_position(p, robotid1[0], robotid1[1], 11, random_pose, random_orient)

    # Find valid robot 2 start pose
    pdu.set_position(p, robotid2[0], robotid2[1], get_random_pose())
    robot2_goal_pose = get_joint_position(p, robotid2[0], robotid2[1])

    random_pose = np.r_[-0.125, 0.5, 0.5]
    random_orient = np.r_[-np.pi, np.pi/2, np.pi]
    set_IK_position(p, robotid2[0], robotid2[1], 11, random_pose, random_orient)

    # Place obstacles
    all_obstacles = set_obstacles(p, 5, 9, 9, robotid1[0], robotid2[0])

    # TODO: Find valid robot 2 start pose
    pdu.set_position(p, robotid1[0], robotid1[1], get_random_pose())
    while pdu.get_distance(p, all_obstacles, robotid1[0])<0:
        pdu.set_position(p, robotid1[0], robotid1[1], get_random_pose())

    # TODO: Find valid robot 1 start pose
    pdu.set_position(p, robotid2[0], robotid2[1], get_random_pose())
    while pdu.get_distance(p, all_obstacles, robotid2[0])<0:
        pdu.set_position(p, robotid2[0], robotid2[1], get_random_pose())
