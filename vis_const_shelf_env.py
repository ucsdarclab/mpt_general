''' Visualize robot and shelf - concept figure
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

import time
import argparse

import panda_utils as pu
import panda_shelf_env as pse

import roboticstoolbox as rtb

if __name__=="__main__":
    p = pu.get_pybullet_server('gui')

    env_num = 1
    # Reset teh environment
    pu.set_simulation_env(p)
    # Place obstacles in space.
    all_obstacles = pse.place_shelf(p, [0.3, -0.6, 0.0], [np.pi/2, 0.0, 0.0])
    # Create a floor for the shelf - 
    # NOTE: Only for visual purposes!
    visualBox = p.createVisualShape(
        pyb.GEOM_BOX, 
        halfExtents=[2.5/2, 2.5/2, 0.1/2], 
        rgbaColor=[0.39, 0.39, 0.39, 1]
    )
    p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visualBox,
            basePosition=[0, 0, -0.075])

    # Load a trajectory and place the robot in random points.
    data_dir = '/root/data/panda_constraint/val'
    with open(osp.join(data_dir, f'env_{env_num:06d}', 'path_0.p'), 'rb') as f:
        path_data = pickle.load(f)
    path = path_data['path_interpolated']
    # goal pose: array([-0.90839, -0.00350605, 0.378341, -1.75229, -1.67308,2.09185, 2.5621]
    # Reset camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.8,
        cameraYaw=94,
        cameraPitch=-31.4,
        cameraTargetPosition=[0., 0., 0.]
    )
    # Keep an object at the end-effector
    panda_model = rtb.models.DH.Panda()
    # Define the cylinderical obstacle
    sph_radius = 0.025
    rgba = [0.125, 0.5, 0.5, 0.6]
    geomCylinder = p.createCollisionShape(pyb.GEOM_CYLINDER, radius=sph_radius, height= 0.2)
    visualCylinder = p.createVisualShape(pyb.GEOM_CYLINDER, radius=sph_radius, length=0.2, rgbaColor=rgba)
    base_pose = panda_model.fkine(path[-1]).t
    shelf_obj = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=geomCylinder,
                    baseVisualShapeIndex=visualCylinder,
                    basePosition=base_pose,
                )
    
    # # Spawn the robot.
    # pandaID, jointsID, fingersID = pu.set_robot(p)
    # pu.set_position(p, pandaID, jointsID, path[-1])
    # # Open gripper.
    # p.resetJointState(pandaID, 9, 0.06/2)
    # p.resetJointState(pandaID, 10, 0.06/2)

    colors = np.array([
            [1, 0, 0, 0.5],
            [0, 1, 0, 0.5],
            [0, 0, 1, 0.5]
        ])
    # TODO: Create axis
    array_directions = np.array([
        [0.1, 0, 0],
        [0, 0.1, 0],
        [0, 0, 0.1],
    ])
    for i, p_i in enumerate(array_directions):
        y_axis = p.addUserDebugLine(lineFromXYZ = base_pose,
                                    lineToXYZ = base_pose + p_i,
                                    lineColorRGB = colors[i][:3],
                                    lineWidth = 4
        )

    # # Randomly place arms near constraints.
    # random_poses = [
    #     [-0.7, -0.00350605, 0.378341, -1.9, -1.67308, 1.5, 2.5621],
    #     [-0.65, -0.00350605, 0.4, -1.65, -1.67308, 2.09185, 2.5621],
    #     [-0.5, -0.00350605, 0.378341, -1.75229, -1.6, 2.09185, 2.5621]
    # ]

    # Randomly place arms away from constraints.
    random_poses = [
        [-0.7, -0.0, 0.5, -1.9, -1.67308, 1.5, 0.0],
        [-0.65, -0.00, 0.4, -1.65, -1.67308, 1.6, 1.5],
        [-0.5, 0.4, 0.3, -1.75229, -1.6, 1.25, -1]
    ]
    
    for i, p_i in enumerate(random_poses):
        panda_vis_ID, joints_vis_ID, _ = pu.set_robot_vis(p, rgbaColor=colors[i])
        pu.set_position(p, panda_vis_ID, joints_vis_ID, p_i)
        p.resetJointState(panda_vis_ID, 9, 0.06/2)
        p.resetJointState(panda_vis_ID, 10, 0.06/2)


