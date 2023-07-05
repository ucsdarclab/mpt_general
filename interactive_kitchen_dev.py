''' Scratch pad for kitchen environment
'''

import pybullet as pyb
import numpy as np
import os
from os import path as osp
import pickle
import time
from functools import partial
from scipy.spatial.transform import Rotation as Rot

from scipy import stats

import panda_utils as pu
from pybullet_object_models import ycb_objects

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    raise "Run code from a container with OMPL installed"

from torch.distributions import MultivariateNormal
from ompl_utils import get_ompl_state, get_numpy_state

import open3d as o3d

# Contact-GraspNet imports!!
import sys
BASE_DIR = '/root/third_party/contact_graspnet'
sys.path.append(osp.join(BASE_DIR, 'contact_graspnet'))

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from contact_grasp_estimator import GraspEstimator
import config_utils
from visualization_utils import visualize_grasps

from interactive_panda_kitchen import *

def add_debug_point(client_id, pose):
    colors = np.zeros_like(pose)
    colors[:, 0] = 1.0
    obj_id = client_id.addUserDebugPoints(pose, colors, 25)
    return obj_id

if __name__=="__main__":
    # Server for collision checking
    p_collision = pu.get_pybullet_server('direct')
    # Server for collision checking
    p_pcd = pu.get_pybullet_server('direct')
    # Server for visualization
    p = pu.get_pybullet_server('gui')
    # p = pu.get_pybullet_server('direct')

    p.resetDebugVisualizerCamera(1.8, -28.6, -33.6, np.array([0.0, 0.0, 0.0]))
    # p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setAdditionalSearchPath(osp.join(os.getcwd(), 'assets'))

    # Set up environment for simulation
    all_obstacles, itm_id = set_env(p)
    kitchen = all_obstacles[0]
    # Set up environment for collision checking
    all_obstacles_coll, itm_id_coll = set_env(p_collision)
    # Set up environment for capturing pcd
    all_obstacles_pcd, itm_id_pcd = set_env(p_pcd)

    # # Open the shelf
    # shelf_index = 29
    # p.resetJointState(all_obstacles[0], shelf_index-2, -1.57)
    # p_collision.resetJointState(all_obstacles_coll[0], shelf_index-2, -1.57)

    # while True:
    #     p.stepSimulation()

    # table = p.loadURDF('table/table.urdf',[1.0,0,0], p.getQuaternionFromEuler([0,0,1.57]), useFixedBase=True)
    # all_joint_names = [
    #     pyb.getJointInfo(kitchen, i)[1].decode('utf-8')
    #     for i in range(pyb.getNumJoints(kitchen))
    # ]
    # handle_ids = [
    #     (i, pyb.getJointInfo(kitchen, i)[1].decode("utf-8")) 
    #     for i in range(pyb.getNumJoints(kitchen))
    #     if 'handle' in pyb.getJointInfo(kitchen, i)[1].decode("utf-8")
    # ]

    # Load the interactive robot
    pandaID, jointsID, fingerID = pu.set_robot(p)
    # Load the collision checking robot
    pandaID_col, jointsID_col, fingerID_col = pu.set_robot(p_collision)
    
    # # Set up the camera:
    view_matrix = pyb.computeViewMatrix(
        # cameraEyePosition=[-0.6, -0.6, 0.8],
        cameraEyePosition=[-0.6, -0.5, 0.7],
        cameraTargetPosition=[0.5, -0.4, 0.15],
        cameraUpVector=[0., 0., 1]
    )
    # # For shelf placing
    # view_matrix = pyb.computeViewMatrix(
    #     cameraEyePosition=[-0.6, -0.4, 1.271],
    #     cameraTargetPosition=[0.5, -0.4, 0.721],
    #     cameraUpVector=[0., 0., 1]
    # )
    fov = 45
    height = 512
    width = 512
    aspect = width/height
    near=0.02
    far=3
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    _, _, rgb_img, depth_img, seg_img = p.getCameraImage(width,
                                height,
                                view_matrix,
                                projection_matrix)
    # ========== Get PCD ===========
    pcd = get_pcd(p_pcd)
    # o3d.visualization.draw_geometries([pcd])
    
    # ================== Trajectory planning =================
    # Define OMPL plannner
    # Planning parameters
    space = ob.RealVectorStateSpace(7)
    bounds = ob.RealVectorBounds(7)
    # Set joint limits
    for i in range(7):
        bounds.setHigh(i, pu.q_max[0, i])
        bounds.setLow(i, pu.q_min[0, i])
    space.setBounds(bounds)
    si = ob.SpaceInformation(space)
    
    # Define collison checker.
    validity_checker_obj = pu.ValidityCheckerDistance(
        p_collision,
        si, 
        all_obstacles_coll,
        pandaID_col,
        jointsID_col
    )
    door_link_index = 29
    
    with open(f'handle_{door_link_index}_traj.pkl', 'rb') as f:
        data = pickle.load(f)
        # Open trajectory
        q_traj = np.array(data['traj'])
    panda_reset_open_gripper(p, pandaID, gripper_dist=0.1)
    panda_reset_open_gripper(p_collision, pandaID_col, gripper_dist=0.1)

    np.random.seed(100)
    # Randomly sample a collision free start point.
    initial_config = (pu.q_min + (pu.q_max-pu.q_min)*np.random.rand(7))[0]
    pu.set_position(p_collision, pandaID_col, jointsID_col, initial_config)

    while pu.get_distance(p_collision, all_obstacles_coll, pandaID_col)<0. or pu.check_self_collision(p_collision, pandaID_col):
        initial_config = (pu.q_min + (pu.q_max-pu.q_min)*np.random.rand(7))[0]
        pu.set_position(p_collision, pandaID_col, jointsID_col, initial_config)
    pu.set_position(p, pandaID, jointsID, initial_config)
    # Plan a trajectory from initial config to cupboard handle grasp location.
    goal_q = q_traj[0]
    traj_cupboard, _, _ , success = get_path(initial_config, goal_q, validity_checker_obj)
    follow_trajectory(p, pandaID, jointsID, traj_cupboard)
    j_c = pu.get_joint_position(p, pandaID, jointsID)
    while np.linalg.norm(j_c-goal_q)<1e-12:
        p.setJointMotorControlArray(
            pandaID,
            jointsID,
            pyb.POSITION_CONTROL,
            targetPositions=goal_q
        )
        p.stepSimulation()
        j_c = pu.get_joint_position(p, pandaID, jointsID)
    
    for _ in range(100):
        p.stepSimulation()
    print(pu.get_joint_position(p, pandaID, jointsID)-goal_q)
    for _ in range(100):
        panda_close_gripper(p, pandaID)
        p.stepSimulation()
    for _ in range(100):
        p.stepSimulation()
    follow_trajectory(p, pandaID, jointsID, q_traj)
    
    # Open panda gripper
    panda_open_gripper(p, pandaID, 0.1)
    p.stepSimulation()
    
    gripper_joint_state = p.getJointState(pandaID, 10)[0]
    print(gripper_joint_state)
    panda_reset_open_gripper(p_collision, pandaID_col, gripper_dist=2*gripper_joint_state)

    # Sync collision env
    cupboard_joint_state = p.getJointState(kitchen, door_link_index-2)[0]
    p_collision.resetJointState(all_obstacles_coll[0], door_link_index-2, cupboard_joint_state)
    # Sync gripper.
    panda_reset_open_gripper(p_collision, pandaID_col, 0.1)

    with open('shelf_reach_q.pkl', 'rb') as f:
        data = pickle.load(f)
        can_start_q = data['start_q']
    with open('shelf_target_q.pkl', 'rb') as f:
        data = pickle.load(f)
        can_goal_q = data['goal_q']

    tmp_start_q = pu.get_joint_position(p, pandaID, jointsID)
    path_cupboard_2_can, _, _, success = get_path(tmp_start_q, can_start_q, validity_checker_obj)
    # Execute cupboard trajectory
    follow_trajectory(p, pandaID, jointsID, path_cupboard_2_can)
    for _ in range(10):
        p.stepSimulation()
    # Plan a trajectory from grasp point to table
    path_can, _, _, success = get_path(can_start_q, can_goal_q, validity_checker_obj)
    # Close the gripper
    for _ in range(200):
        panda_close_gripper(p, pandaID)
        p.stepSimulation()
    follow_trajectory(p, pandaID, jointsID, path_can)
    
    # # ================ Grasping code. ======================
    # vM = np.reshape(view_matrix, (4, 4), order='F')
    # pM = np.reshape(projection_matrix, (4, 4), order='F')
    # tran_pix_world = np.linalg.inv(pM@vM)
    
    # # Get point cloud data from depth.
    # pcd = depth2pc_opengl(depth_img, tran_pix_world, height, width)
    
    # # Contact-GraspNet chkpt
    # ckpt_dir = '/root/third_party/models/scene_test_2048_bs3_hor_sigma_0025'
    # forward_passes = 5
    # global_config = config_utils.load_config(ckpt_dir, batch_size=forward_passes)
    # # Build the model
    # grasp_estimator = GraspEstimator(global_config)
    # grasp_estimator.build_network()
    # # Add ops to save and restore all the variables.
    # saver = tf.train.Saver(save_relative_paths=True)
    # # Create a session
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    # sess = tf.Session(config=config)
    # # Load weights
    # grasp_estimator.load_weights(sess, saver, ckpt_dir, mode="test")
    # # NOTE: Produced grasps are in camera frame!!

    # # Translate pcd (world frame) to camera frame
    # view_pcd = (np.c_[pcd, np.ones(pcd.shape[0])]@vM.T)[:, :3]
    # # Rotate OpenGL co-ordinates to CV format
    # R_VM_2_CV = np.array([
    #     [1, 0, 0],
    #     [0, -1.0, 0],
    #     [0, 0, -1.0]
    # ])
    # cv_pcd = view_pcd@R_VM_2_CV.T
    # pc_segmap = seg_img.ravel(order='C')
    # pc_segments = {}
    # for i in np.unique(pc_segmap):
    #     if i == 0:
    #         continue
    #     else:
    #         pc_segments[i] = cv_pcd[pc_segmap == i]
    # # Predict grasping points
    # pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(
    #     sess,
    #     cv_pcd,
    #     pc_segments=pc_segments,
    #     local_regions=True,
    #     filter_grasps=True,
    #     forward_passes=forward_passes,
    # )
    # pc_colors = rgb_img[:, :, :3].reshape(-1, 3)
    # # Rotate along z-axis the grasp orientation (in the grasp frame)
    # # by 180deg.
    # inverted_pred_grasps = {}
    # inv_T = np.eye(4)
    # inv_T[0, 0] = -1.0
    # inv_T[1, 1] = -1.0
    # for key in pred_grasps_cam.keys():
    #     inverted_pred_grasps[key] = pred_grasps_cam[key]@inv_T[None, :, :]
    
    # # Visualize PC and grasping data.
    # # visualize_grasps(cv_pcd, pred_grasps_cam, scores, pc_colors=pc_colors, plot_opencv_cam=True)
    # visualize_grasps(cv_pcd, inverted_pred_grasps, scores, pc_colors=pc_colors, plot_opencv_cam=True)
    
    # # ========== Find a grasp for which an IK exists ========================
    # panda_reset_open_gripper(pandaID, gripper_dist=0.1)
    # grasp_index_asc = np.argsort(scores[3])[::-1]
    # for grasp_index in grasp_index_asc:
    #     # OpenCV to grasp 

    #     T_CV_G = pred_grasps_cam[3][grasp_index]
    #     # Camera co-ordinate to grasp
    #     T_C_G = np.r_[R_VM_2_CV.T@T_CV_G[:3, :], np.array([[0, 0, 0, 1.0]])]
    #     # In future use better ways to invert transformation matrix.
    #     T_W_G = np.linalg.inv(vM)@T_C_G

    #     grasp_orient = Rot.from_matrix(T_W_G[:3, :3]).as_quat()
    #     # Find IK for the given tranformation:
    #     # q, solved = get_IK_pose(pandaID, jointsID, ee_pose=T_W_G[:3, 3], ee_orient=grasp_orient, link_id=8)
    #     q, solved = get_IK_posev2(pandaID, jointsID, T_W_G)
    #     if solved:
    #         break
    #     # Check if grasp exists for inverted orientation
    #     print("Checking for inverted grasp")
    #     # OpenCV to grasp 
    #     T_CV_G = inverted_pred_grasps[3][grasp_index]
    #     # Camera co-ordinate to grasp
    #     T_C_G = np.r_[R_VM_2_CV.T@T_CV_G[:3, :], np.array([[0, 0, 0, 1.0]])]
    #     # In future use better ways to invert transformation matrix.
    #     T_W_G = np.linalg.inv(vM)@T_C_G
        
    #     grasp_orient = Rot.from_matrix(T_W_G[:3, :3]).as_quat()
    #     # Find IK for the given tranformation:
    #     # q, solved = get_IK_pose(pandaID, jointsID, ee_pose=T_W_G[:3, 3], ee_orient=grasp_orient, link_id=8)
    #     q, solved = get_IK_posev2(pandaID, jointsID, T_W_G)
    #     if solved:
    #         break
    # ========================================================================
    # Find the "BEST" grasp with the highest scores for objects on the shelf:
    # grasp_index = np.argmax(scores[3]) # Index 3 is the chips can
    # T_CV_G = pred_grasps_cam[3][grasp_index]
    # T_C_G = np.r_[R_VM_2_CV.T@T_CV_G[:3, :], np.array([[0, 0, 0, 1.0]])]
    # # In future use better ways to invert transformation matrix.
    # T_W_G = np.linalg.inv(vM)@T_C_G
    
    # # Find IK for the given tranformation:
    # panda_reset_open_gripper(pandaID, gripper_dist=0.1)
    # grasp_orient = Rot.from_matrix(T_W_G[:3, :3]).as_quat()
        # solved = False
    # while not solved:
    #     q, solved = get_IK_pose(pandaID, jointsID, ee_pose=T_W_G[:3, 3], ee_orient=grasp_orient, link_id=8)
    # ============================== best grasp ==============================

    # ---------------------------- end-of-grasping --------------------------------
    
    # # Save data for testing.
    # with open('depth_img.pkl', 'wb') as f:
    #     pickle.dump(
    #     {
    #         'rbg_img': rgb_img,
    #         'depth_img': depth_img,
    #         'seg_img': seg_img,
    #         'near': near,
    #         'far': far
    #     }, f)
    # # Identify grasping locations:
    # while True:
    #     p.stepSimulation()
    #============================ Trajectories for opening/closing the shelf ===================
    # # Identify grasping locations within the scene
    # # NOTE: Left/Right are viwed after the kitchen is rotated by 180 around z-axis.
    # # Hence there may be a discrepancy w/ the name of the joint and definitions here
    # # Left draw handles - 39(top), 42(bottom)
    # # Left top cupboards - 20(right), 24(middle), 16 (left)
    # # Right draw handles - 57(top), 59(bottom)
    # # Right top cupboards - 33(right), 29 (left)

    # construct a trajectory for the draw
    link_index = 29
    
    # shelf_traj = []
    # shelf_orient = []
    # # panda_reset_close_gripper(pandaID)
    # # # For drawer
    # # for i in np.linspace(0, 0.35, 20):
    # #     link_joint_index = link_index-1
    # # # For cupboard
    # # for i in np.linspace(-1.57, 0, 40)[::-1]:
    # shelf_joint_traj = np.linspace(-np.pi*0.45, 0, 80)[::-1]
    # for i in shelf_joint_traj:
    #     link_joint_index = link_index-2
    #     p.resetJointState(kitchen, link_joint_index, i)
    #     shelf_traj.append(np.array(p.getLinkState(kitchen, link_index)[0]))
    #     shelf_orient.append(np.array(p.getLinkState(kitchen, link_index)[1]))
    # shelf_traj = np.array(shelf_traj)
    # shelf_orient = np.array(shelf_orient)
    # p.resetJointState(kitchen, link_joint_index, 0.0)
    # # # NOTE: Running sim helps with grasping stability!!
    # for _ in range(10):
    #     p.stepSimulation()
    # # choose a handle pose that the robot can reach.
    # handle_pose = np.array(p.getLinkState(kitchen, link_index)[0])
    # # # add_debug_point(p, handle_pose[None, :])
    # handle_orient = [0.0, np.pi/2, 0.0]
    # handle_orient_q = pyb.getQuaternionFromEuler(handle_orient)
    # # # For draw
    # # R_H_T = np.array(
    # #     [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    # # )
    # # For cabinet
    # R_H_T = np.array([
    #     [0, 0, -1], [0, -1, 0], [-1, 0, 0]
    # ])
    # inv_R = np.eye(3)
    # inv_R[0, 0] = -1.0
    # inv_R[1, 1] = -1.0
    # R_handle = Rot.from_quat(pyb.getLinkState(kitchen, link_index)[1])
    # # R_target = R_handle.as_matrix()@R_H_T  # Normal grasp
    # R_target = R_handle.as_matrix()@inv_R@R_H_T # Inverted grasp
    # handle_orient_q = Rot.from_matrix(R_target).as_quat()
    # handle_pose = handle_pose + np.array([-0.2, 0.0, 0.])
    # panda_reset_open_gripper(p, pandaID, 0.025)
    # # find the joint angles that will open the draw
    # q_traj = []
    # q_i = None
    # for i, handle_pose in enumerate(shelf_traj):
    #     R_handle = Rot.from_quat(shelf_orient[i])
    #     R_target = R_handle.as_matrix()@R_H_T
    #     T_target = np.eye(4)
    #     T_target[:3, :3] = R_target
    #     # Offset grasping point by -0.2 if using TracIK Solver.
    #     T_target[:3, 3] = handle_pose + R_target@np.array([0., 0.0, -0.1]).T
    #     handle_orient_q = Rot.from_matrix(R_target).as_quat()
    #     q, _ = get_IK_posev2(p, pandaID, jointsID, T_target, q_i)
    #     # set q_i for the first joint value.
    #     if q_i is None:
    #         q_i = q
    #     # q, _ = get_IK_pose(p, pandaID, jointsID, handle_pose, handle_orient_q, q_i)
    #     # Check if robot is in collision w/ itself or kitchen and close to the previous solution.
    #     not_reached=False
    #     not_reached = True
    #     # print(f"Joint distance : {np.linalg.norm(q_i-q)}"+ f" Cartesian distance : {np.linalg.norm(np.array(p.getLinkState(pandaID, 11)[0])-handle_pose)}")
    #     if np.linalg.norm(np.array(p.getLinkState(pandaID, 11)[0])-handle_pose)<1e-2:
    #         not_reached = False
    #     while pu.get_distance(p, all_obstacles, pandaID)<0. or pu.check_self_collision(p, pandaID) or not_reached:
    #         # q, _ = get_IK_pose(p, pandaID, jointsID, handle_pose, handle_orient_q, q_i)
    #         # q, _ = get_IK_posev2(p, pandaID, jointsID, T_target, q_i)
    #         q, _ = get_IK_posev2(p, pandaID, jointsID, T_target, None)
    #         not_reached = True
    #         # print(f"Joint distance : {np.linalg.norm(q_i-q)}"+ f" Cartesian distance : {np.linalg.norm(np.array(p.getLinkState(pandaID, 11)[0])-handle_pose)}")
    #         if np.linalg.norm(np.array(p.getLinkState(pandaID, 11)[0])-handle_pose)<1e-2:
    #             not_reached = False
    #     print(np.linalg.norm(np.array(p.getLinkState(pandaID, 11)[0])-handle_pose))
    #     q_i = q
    #     q_traj.append(np.array(q)[:7])
    # pu.set_position(p, pandaID, jointsID, q_traj[0])
    # for q_i in q_traj:
    #     pu.set_position(p, pandaID, jointsID, q_i)
    #     time.sleep(0.1)
    # with open(f'handle_{link_index}_traj.pkl', 'wb') as f:
    #     pickle.dump({'traj':q_traj}, f)
    # pyb.resetJointState(kitchen, link_joint_index, 0.35)
    # with open(f'handle_{link_index}_traj.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # # Open trajectory
    # q_traj = np.array(data['traj'])
    
    # # panda_reset_open_gripper(p, pandaID)
    # # Place the arm at the handle
    # pu.set_position(p, pandaID, jointsID, q_traj[0])
    # for _ in range(100):
    #     panda_close_gripper(p, pandaID)
    #     p.stepSimulation()
    # follow_trajectory(p, pandaID, jointsID, q_traj)
    
    # Identify re-grasping points.
    # dis_index = np.where(np.linalg.norm(np.diff(q_traj, axis=0), axis=1)>1)[0]

    # # connect_traj, _, _, success = get_path(q_traj[dis_index], q_traj[dis_index+1], validity_checker_obj)
    # # connect_traj = np.array()
    # # # Close trajectory
    # # q_traj = data['traj'][::-1]
    # # # execute the trajectory
    # # # ------------- Move the robot ------------------------------------
    # panda_reset_open_gripper(p, pandaID)
    # # Place the arm at the handle
    # pu.set_position(p, pandaID, jointsID, q_traj[0])
    # for start_index, stop_index in zip(np.r_[0, dis_index+1], np.r_[dis_index, q_traj.shape[0]]):
    #     for _ in range(100):
    #         panda_close_gripper(p, pandaID)
    #         p.stepSimulation()
    #     # for q_i in q_traj[start_index:stop_index]:
    #     #     pu.set_position(p, pandaID, jointsID, q_i)
    #     #     time.sleep(0.1)
    #     follow_trajectory(p, pandaID, jointsID, q_traj[start_index:stop_index])
    #     # Open end-effector links
    #     for _ in range(10):
    #         panda_open_gripper(p, pandaID, 0.1)
    #         p.stepSimulation()
    #     # Open end-effector links
    #     # panda_reset_open_gripper(p, pandaID, 0.025)
    #     panda_reset_open_gripper(p_collision, pandaID_col, 0.1)
    #     # Sync cupboard state.
    #     cupboard_joint_state = p.getJointState(kitchen, link_index-2)[0]
    #     p_collision.resetJointState(all_obstacles_coll[0], link_index-2, cupboard_joint_state)
    #     if stop_index<q_traj.shape[0]:
    #         connect_traj, _, _, success = get_path(q_traj[stop_index], q_traj[stop_index+1], validity_checker_obj)
    #         follow_trajectory(p, pandaID, jointsID, connect_traj)
