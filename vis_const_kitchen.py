''' Visualizer for the kitchen environment - trajectory
'''

import pybullet as pyb
import numpy as np
import os
from os import path as osp
import pickle
import time

import panda_utils as pu
from pybullet_object_models import ycb_objects

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    raise "Run code from a container with OMPL installed"

import torch
from torch.distributions import MultivariateNormal


import json
import spatialmath as spm
import roboticstoolbox as rtb

import eval_const_7d as ec7
import interactive_panda_kitchen as ipk
import interactive_kitchen_dev as ikd
import panda_constraint_shelf as pcs


if __name__ == "__main__":
    # Server for visualization/execution
    p = pu.get_pybullet_server('gui')
    # Reset camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.1,
        cameraYaw=-46.00,
        cameraPitch=-29.80,
        cameraTargetPosition=[0.28, -0.28, 0.62]
    )

    p.setAdditionalSearchPath(osp.join(os.getcwd(), 'assets'))
    env_num = 1

    # Set up environment for simulation
    all_obstacles, itm_id = ipk.set_env(p, seed=env_num)
    kitchen = all_obstacles[0]

    # Define panda model using roboticstoolbox.
    panda_model = rtb.models.DH.Panda()

    # # Load the interactive robot
    # pandaID, jointsID, fingerID = pu.set_robot(p)
    # ipk.panda_reset_open_gripper(p, pandaID, gripper_dist=0.08)

    # Open the shelf
    shelf_index = 29
    p.resetJointState(all_obstacles[0], shelf_index-2, -1.57)

    with open(f'cvq_mpt_{env_num}.p', 'rb') as f:
        data = pickle.load(f)
    path = data['path']

    # Find the initial difference b/w the poses.
    obj_pose, obj_w = p.getBasePositionAndOrientation(itm_id)
    # Orientation is I
    T_r_o = spm.SE3.Trans(obj_pose[0], obj_pose[1], obj_pose[2])
    T_r_ee = panda_model.fkine(path[-1])
    # T_o_offset = spm.SE3.Trans(0., 0, 0.03) # For video
    T_o_offset = spm.SE3.Trans(0., 0, -0.03)
    T_ee_o = T_r_ee.inv()@T_r_o@T_o_offset
    # # ================== For video ======================================
    # time.sleep(10)
    # for q_i in path[::-1]:
    #     # TODO: Move the chips can with respect to the arm
    #     T_r_ee = panda_model.fkine(q_i)
    #     T_r_o = T_r_ee@T_ee_o
    #     p.resetBasePositionAndOrientation(itm_id, T_r_o.t, spm.base.r2q(T_r_o.R, order='xyzs'))
    #     pu.set_position(p, pandaID, jointsID, q_i)

    #     time.sleep(0.1)
    # # ====================================================================

    # ============================= Static Figure ==========================
    p.removeBody(itm_id)
    obj_name = 'YcbChipsCan'
    path_to_urdf = osp.join(ycb_objects.getDataPath(), obj_name, "model.urdf")
    # Add start panda
    # Add a panda robot.
    pandaID_vis, jointsID_vis, fingerID_vis = pu.set_robot_vis(p, rgbaColor=[1, 0, 0, 0.7])
    pu.set_position(p, pandaID_vis, jointsID_vis, path[0])
    ipk.panda_reset_open_gripper(p, pandaID_vis, gripper_dist=0.08)
    
    # Place the obstacle in the space.
    T_r_ee = panda_model.fkine(path[0])
    T_r_o = T_r_ee@T_ee_o
    itm_id = p.loadURDF(
        path_to_urdf, 
        basePosition=T_r_o.t,
        baseOrientation=spm.base.r2q(T_r_o.R, order='xyzs')
        )
    # Add goal panda
    # Add a panda robot.
    pandaID_vis, jointsID_vis, fingerID_vis = pu.set_robot_vis(p, rgbaColor=[0, 1, 0, 0.7])
    pu.set_position(p, pandaID_vis, jointsID_vis, path[-1])
    ipk.panda_reset_open_gripper(p, pandaID_vis, gripper_dist=0.08)
    
    # Place the obstacle in the space.
    T_r_ee = panda_model.fkine(path[-1])
    T_r_o = T_r_ee@T_ee_o
    itm_id = p.loadURDF(
        path_to_urdf, 
        basePosition=T_r_o.t,
        baseOrientation=spm.base.r2q(T_r_o.R, order='xyzs')
        )
    for q_i in path[[ 90, 80], :]:
        # Add a panda robot.
        pandaID_vis, jointsID_vis, fingerID_vis = pu.set_robot_vis(p, rgbaColor=[1, 1, 1, 0.6])
        pu.set_position(p, pandaID_vis, jointsID_vis, q_i)
        ipk.panda_reset_open_gripper(p, pandaID_vis, gripper_dist=0.08)
        
        # TODO: Place the obstacle in the space.
        T_r_ee = panda_model.fkine(q_i)
        T_r_o = T_r_ee@T_ee_o
        itm_id = p.loadURDF(
            path_to_urdf, 
            basePosition=T_r_o.t,
            baseOrientation=spm.base.r2q(T_r_o.R, order='xyzs')
            )
    # ======================================================================