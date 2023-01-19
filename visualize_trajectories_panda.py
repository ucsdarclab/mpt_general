''' Visualize robot trajectories
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

from panda_utils import q_max, q_min, get_pybullet_server
from panda_utils import set_simulation_env, set_obstacles, set_robot

def set_position(model, joints, jointValue):
    ''' Set the model robot to the given joint values
    :param model: pybullet id of link.
    :param jointValue: joint value to be set
    '''
    for jV, j in zip(jointValue, joints):
        pyb.resetJointState(model, j, jV)

def set_visual_env(client_obj, num_boxes, num_spheres, seed):
    '''
    Generate environment with randomly placed obstacles in space.
    :param client_obj: bc.BulletClient object
    :param num_boxes:
    :param num_spheres:
    :param seed:
    :returns ValidityCheckerObj:
    '''
    set_simulation_env(client_obj)
    panda, joints, _ = set_robot(client_obj)
    obstacles = set_obstacles(client_obj, num_boxes=num_boxes, num_spheres=num_spheres, seed = seed)
    return panda, joints, obstacles

if __name__ == "__main__":
    # TODO: Load joint trajectory.
    env_num = 1133
    path_num = 0

    p = get_pybullet_server('gui')

    # # =================== Place cupboard =================================
    # # Spawn the robot.
    # pandaID, jointsID, _ = set_robot(p)
    # # Place the desk at a range of -0.8:-0.6 meters
    # shift = [0.0, -0.6, 0.0]
    # visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
    #                                 fileName="assets/cupboard.obj",
    #                                 rgbaColor=[1, 1, 1, 1],
    #                                 specularColor=[0.4, .4, 0],
    #                                 )
    # collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
    #                                     fileName="assets/cupboard.obj",
    #                                     )
    # base_orientation = pyb.getQuaternionFromEuler([np.pi/2, 0.0, 0.0])
    # obstacle_cupboard = p.createMultiBody(baseMass=0.0,
    #                 #   baseInertialFramePosition=shift,
    #                   baseCollisionShapeIndex=collisionShapeId,
    #                   baseVisualShapeIndex=visualShapeId,
    #                   basePosition=shift,
    #                   baseOrientation=base_orientation,
    #                   useMaximalCoordinates=True
    #                   )

    # # TODO: Place objects in space.

    # # TODO: Write IK for place location.

    # # TODO: Find path
    # # ==================================================================

    model_folder = '/root/data/general_mpt/stage2/model6'
    # planner_type = 'rrtstar'
    # with open(osp.join(model_folder, f'eval_val_plan_{planner_type}_{2000:06d}.p'), 'rb') as f:
    #     eval_data = pickle.load(f)
    # index_num = env_num - 2000
    # path = eval_data['Path'][index_num]
    # success = eval_data['Success'][index_num]
    data_folder = '/root/data/pandav3/train'
    # path_num = 2
    # with open(osp.join(data_folder, f'env_{env_num:06d}', f'path_{path_num}.p'), 'rb') as f:
    #     data = pickle.load(f)
    # path = data['jointPath']
    panda, joints, obstacles = set_visual_env(p, 6, 6, seed=env_num)
    # alpha = np.linspace(0, 1, 10)[:, None]
    # for i, _ in enumerate(path[:-1]):
    #     # Interpolate b/w joints.
    #     tmp_path = (1-alpha)*path[i] + alpha*path[i+1]
    #     for pos in tmp_path:
    #         # Set robot position.
    #         set_position(panda, joints, pos)
    #         time.sleep(0.1)    
    # if success:
    #     # Calculate the FK of the arm.        
    #     panda, joints, obstacles = set_visual_env(p, 6, 6, seed=env_num)
    #     alpha = np.linspace(0, 1, 10)[:, None]
    #     for i, _ in enumerate(path[:-1]):
    #         # Interpolate b/w joints.
    #         tmp_path = (1-alpha)*path[i] + alpha*path[i+1]
    #         for pos in tmp_path:
    #             # Set robot position.
    #             set_position(panda, joints, pos)
    #             time.sleep(0.1)
    # else:
    #     print("Couldn't find a successful path")

    # # TODO: Add to the scene.
