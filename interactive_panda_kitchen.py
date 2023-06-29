''' An interactive kitchen enviornment with the panda robot.
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

def add_debug_point(client_id, pose):
    colors = np.zeros_like(pose)
    colors[:, 0] = 1.0
    obj_id = client_id.addUserDebugPoints(pose, colors, 25)
    return obj_id

def panda_reset_open_gripper(robotID, gripper_dist=0.02):
    '''
    Open the grippers for the panda robot.
    :param robotID: pybullet robot id for the panda arm.
    :param gripper_dis: distance between the gripper.
    '''
    pyb.resetJointState(robotID, 9, gripper_dist/2)
    pyb.resetJointState(robotID, 10, gripper_dist/2)

def panda_reset_close_gripper(robotID):
    '''
    Close the grippers for the panda robot.
    :param robotID: pybullet robot id for the panda arm.
    '''
    pyb.resetJointState(robotID, 9, 0.0)
    pyb.resetJointState(robotID, 10, 0.0)

def panda_close_gripper(clientID, robotID):
    '''
    Close the gripper using velocity control.
    :param clientID: pybullet client id.
    :param robotID: pybullet robot id for the panda arm.
    '''
    clientID.setJointMotorControlArray(
        robotID,
        [9, 10],
        pyb.VELOCITY_CONTROL,
        targetVelocities=-np.ones(2)*0.05
    )

def get_IK_pose(robotID, jointsID, ee_pose, ee_orient, init_q=None, link_id=11):
    '''
    Find the robot joint values for the given end-effector pose.
    :param robotID:
    :param jointsID:
    :param ee_pose:
    :param ee_orient:
    '''
    if init_q is None:
        init_q = (pu.q_min+np.random.rand(7)*(pu.q_max-pu.q_min))[0]
    q = init_q
    pu.set_position(robotID, jointsID, q)
    attempt = 0
    while np.linalg.norm(np.array(pyb.getLinkState(robotID, link_id)[0])-ee_pose)>1e-4:
        q = p.calculateInverseKinematics(
            robotID,
            link_id,
            ee_pose,
            ee_orient,
            maxNumIterations=75
        )
        pu.set_position(robotID, jointsID, q)
        attempt +=1
        if attempt>5:
            return q, False
    return q, True

from tracikpy import TracIKSolver

def get_IK_posev2(robotID, jointsID, eeT):
    '''
    Find the robot joint values for the given end-effector pose using
    TracIKSolver.
    :param robotID: 
    :param jointsID:
    :param eeT:
    '''
    ik_solver = TracIKSolver(
        "assets/franka_panda/franka_panda.urdf",
        "panda_link0",
        "panda_hand"
    )
    qout = ik_solver.ik(eeT)
    if qout is None:
        qout = (pu.q_min+np.random.rand(7)*(pu.q_max-pu.q_min))[0]
        return qout, False
    pu.set_position(robotID, jointsID, qout)
    return qout, True

GRIPPER_LENGTH = 0.1

def depth2pc_opengl(depth_img, tran_pix_world, height, width):
    '''
    Translate depth img to point cloud for OpenGL data.
    From - https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
    :param depth_img: depth image from image capture.
    :param tran_pix_world: Trnasformation for going from pixel co-ordinate to world co-ordinate
    :param height: height of the image.
    :param width: widht of the image.
    :returns np.array: Array of data, forming the point cloud.
    '''
    pcd = []
    for h in range(0, height):
        for w in range(0, width):
            # Removes points that are outliers
            if depth_img[h, w]<1:
                x = (2*w - width)/width
                y = -(2*h - height)/height  # be careful！ depth and its corresponding position
                z = 2*depth_img[h,w] - 1
                pixPos = np.asarray([x, y, z, 1])
                position = np.matmul(tran_pix_world, pixPos)
                # if position[3]!=1:
                #     print(position)
                position = position/position[3]
                pcd.append(position[:3])
    return np.array(pcd)


class StateSamplerRegion(ob.StateSampler):
    '''A class to sample robot joints from a given joint configuration.
    '''
    def __init__(self, space, qMin=None, qMax=None, dist_mu=None, dist_sigma=None):
        '''
        :param space:
        :param qMin: np.array of minimum joint bound
        :param qMax: np.array of maximum joint bound
        :param region: np.array of points to sample from
        '''
        super(StateSamplerRegion, self).__init__(space)
        self.name_ ='region'
        self.q_min = qMin
        self.q_max = qMax
        if dist_mu is None:
            self.X = None
            self.U = stats.uniform(np.zeros_like(qMin), np.ones_like(qMax))
        else:
            # self.X = MultivariateNormal(dist_mu,torch.diag_embed(dist_sigma))
            self.seq_num = dist_mu.shape[0]
            self.X = MultivariateNormal(dist_mu, dist_sigma)

                       
    def get_random_samples(self):
        '''Generates a random sample from the list of points
        '''
        index = 0
        random_samples = np.random.permutation(self.X.sample()*(self.q_max-self.q_min)+self.q_min)
        random_samples[:, 6] = 0.0
        # random_samples[:, 6] = 1.9891

        while True:
            yield random_samples[index, :]
            index += 1
            if index==self.seq_num:
                random_samples = np.random.permutation(self.X.sample()*(self.q_max-self.q_min)+self.q_min)
                random_samples[:, 6] = 0.0
                # random_samples[:, 6] = 1.9891
                index = 0
                
    def sampleUniform(self, state):
        '''Generate a sample from uniform distribution or key-points
        :param state: ompl.base.Space object
        '''
        if self.X is None:
            sample_pos = ((self.q_max-self.q_min)*self.U.rvs()+self.q_min)[0]
            sample_pos[6] = 0.0
            # sample_pos[6] = 1.9891
        else:
            sample_pos = next(self.get_random_samples())
        for i, val in enumerate(sample_pos):
            state[i] = float(val)
        return True


def get_path(start, goal, validity_checker_obj, dist_mu=None, dist_sigma=None):
    '''
    Plan a path given the start, goal and patch_map.
    :param start:
    :param goal:
    :param validity_checker_obj:
    :param dist_mu:
    :param dist_sigma:
    returns (list, float, int, bool): Returns True if a path was planned successfully.
    '''
    # Planning parameters
    space = ob.RealVectorStateSpace(7)
    bounds = ob.RealVectorBounds(7)
    # Set joint limits
    for i in range(7):
        bounds.setHigh(i, pu.q_max[0, i])
        bounds.setLow(i, pu.q_min[0, i])
    space.setBounds(bounds)

    # Redo the state sampler
    state_sampler = partial(StateSamplerRegion, dist_mu=dist_mu, dist_sigma=dist_sigma, qMin=pu.q_min, qMax=pu.q_max)
    space.setStateSamplerAllocator(ob.StateSamplerAllocator(state_sampler))

    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(validity_checker_obj)

    start_state = get_ompl_state(space, start)
    goal_state = get_ompl_state(space, goal)

    success = False

    # Define planning problem
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_state, goal_state)

    # Define planner
    planner = og.RRTConnect(si)
    planner.setRange(13)

    # Set the problem instance the planner has to solve
    planner.setProblemDefinition(pdef)
    planner.setup()

    # Attempt to solve the planning problem in the given time
    start_time = time.time()
    solved = planner.solve(150.0)
    current_time = 0.0
    plan_time = time.time()-start_time
    # Get planner data
    plannerData = ob.PlannerData(si)
    planner.getPlannerData(plannerData)
    numVertices = plannerData.numVertices()

    if pdef.hasExactSolution():
        success = True
        # Simplify solution
        path_simplifier = og.PathSimplifier(si)
        try:
            path_simplifier.simplify(pdef.getSolutionPath(), 0.0)
        except TypeError:
            print("Path not able to simplify for unknown reason!")
            pass
        print("Found Solution")
        pdef.getSolutionPath().interpolate()
        # Get final path. 
        path = [
            get_numpy_state(pdef.getSolutionPath().getState(i))
            for i in range(pdef.getSolutionPath().getStateCount())
            ]
        print(f"Path length after path simplification: {pdef.getSolutionPath().length()}")
    else:
        path = [start, goal]
    return path, plan_time, numVertices, success


def follow_trajectory(client_id, robotID, jointsID, q_traj):
    '''
    :param client_id: bullet client object.
    :param robotID: pybullet id for robot.
    :param jointsID: pybullet robot links to control
    :param q_traj: The trajectory to follow:
    '''
    for q_i in q_traj[:]:
        # Get the current joint state
        j_c = np.array(list(
            map(lambda x: x[0], client_id.getJointStates(robotID, jointsID))
        ))
        count = 0
        # Apply control till robot reaches final goal, or after 10
        # position updates
        while np.linalg.norm(j_c-q_i)>1e-2 and count<10:
            p.setJointMotorControlArray(
                robotID,
                jointsID,
                pyb.POSITION_CONTROL,
                targetPositions=q_i
            )
            client_id.stepSimulation()
            time.sleep(0.1)
            # Update current position
            j_c = np.array(list(
                map(lambda x: x[0], client_id.getJointStates(robotID, jointsID))
            ))
            count += 1

def set_env(client_id):
    '''
    Set up the environment for the given client.
    :param client_id: pybullet client object
    '''
    # Load the kitchen envionment
    offset = 0.75
    # Define robot base
    rb_geom_box = client_id.createCollisionShape(pyb.GEOM_BOX, halfExtents=[0.25/2, 0.25/2, offset/2])
    rb_visual_box = client_id.createVisualShape(pyb.GEOM_BOX, halfExtents=[0.25/2, 0.25/2, offset/2], rgbaColor=[1, 1, 1, 1])
    robot_base = client_id.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=rb_visual_box,
        baseVisualShapeIndex=rb_geom_box,
        basePosition=[-0.05, 0.0, -0.01-offset/2]
    )
    kitchen_path = 'kitchen_description/urdf/kitchen_part_right_gen_convex.urdf'
    kitchen_ori = pyb.getQuaternionFromEuler([0.0, 0.0, np.pi])
    kitchen = client_id.loadURDF(kitchen_path, [0.5, 0.675, 1.477-offset], baseOrientation=kitchen_ori, useFixedBase=True)
    floor = client_id.loadURDF('floor/floor.urdf', [0, 0, -offset], useFixedBase=True)
    all_obstacles = [kitchen, robot_base]

    # Obstacles on the table:
    obj_name = 'YcbChipsCan' # Pick the object up and place it in the draw.
    # obj_name = 'YcbCrackerBox'  # Pick the object up and place it in the shelf.
    # obj_name = 'YcbMasterChefCan'
    # obj_name = 'YcbPottedMeatCan'
    # obj_name = 'YcbMustardBottle'
    path_to_urdf = osp.join(ycb_objects.getDataPath(), obj_name, "model.urdf")
    # # Goal position of the object
    # itm_id = p.loadURDF(path_to_urdf, [0.3, -0.5, 0.171])
    # Start postion of the object
    itm_id = client_id.loadURDF(path_to_urdf, [0.555, -0.385, 0.642])
    return all_obstacles, itm_id



if __name__=="__main__":
    # Server for collision checking
    p_collision = pu.get_pybullet_server('direct')
    # Server for visualization
    p = pu.get_pybullet_server('gui')
    # p = pu.get_pybullet_server('direct')

    p.resetDebugVisualizerCamera(1.8, -28.6, -33.6, np.array([0.0, 0.0, 0.0]))
    # p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setAdditionalSearchPath(osp.join(os.getcwd(), 'assets'))

    # Set up environment for simulation
    all_obstacles, itm_id = set_env(p)
    # Set uip environment for collision checking
    all_obstacles_coll, itm_id_coll = set_env(p_collision)

    # Open the shelf
    shelf_index = 29
    p.resetJointState(all_obstacles[0], shelf_index-2, -1.57)
    p_collision.resetJointState(all_obstacles_coll[0], shelf_index-2, -1.57)

    # while True:
    #     p.stepSimulation()

    # table = p.loadURDF('table/table.urdf',[1.0,0,0], p.getQuaternionFromEuler([0,0,1.57]), useFixedBase=True)
    # drawer_to_joint_id = {
    #     1: 18, 
    #     2: 22, 
    #     3: 27, 
    #     4: 31,
    #     5: 37, 
    #     6: 40, 
    #     7: 48, 
    #     8: 53, 
    #     9: 56, 
    #     10: 58, 
    #     11: 14
    # }
    # drawer_to_joint_limits = {
    #     1: (0, 1.57), 
    #     2: (-1.57, 0), 
    #     3: (-1.57, 0), 
    #     4: (0, 1.57),
    #     5: (0.0, 0.4), 
    #     6: (0.0, 0.4), 
    #     7: (0, 1.57), 
    #     8: (-1.57, 0), 
    #     9: (0.0, 0.4), 
    #     10: (0.0, 0.4), 
    #     11: (0, 1.57)
    # }
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
    
    # ================== Trajectory planning =================
    with open('shelf_reach_q.pkl', 'rb') as f:
        data = pickle.load(f)
        can_start_q = data['start_q']
    with open('shelf_target_q.pkl', 'rb') as f:
        data = pickle.load(f)
        can_goal_q = data['goal_q']

    # Open panda gripper - 
    panda_reset_open_gripper(pandaID, gripper_dist=0.1)

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
    # Plan a trajectory from handle grasp to can
    with open(f'handle_{shelf_index}_traj.pkl', 'rb') as f:
        data = pickle.load(f)
        cupboard_open_traj = np.array(data['traj'])
        tmp_start_q = cupboard_open_traj[-1]
        tmp_start_q[6] = tmp_start_q[6]-2*np.pi
    path_cupboard_2_can, _, _, success = get_path(tmp_start_q, can_start_q, validity_checker_obj)
    pu.set_position(p, pandaID, jointsID, path_cupboard_2_can[0])
    follow_trajectory(p, pandaID, jointsID, path_cupboard_2_can)

    # Plan a trajectory from grasp point to table
    path_can, _, _, success = get_path(can_start_q, can_goal_q, validity_checker_obj)
    # pu.set_position(pandaID, jointsID, path_can[0])
    # Close the gripper
    for _ in range(200):
        panda_close_gripper(p, pandaID)
        p.stepSimulation()
    follow_trajectory(p, pandaID, jointsID, path_can)
    # for p_i in path:
    #     pu.set_position(pandaID, jointsID, p_i)
    #     time.sleep(0.1)
    
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

    # # construct a trajectory for the draw
    # # open draw trajectory
    # link_index = 29
    
    # shelf_traj = []
    # shelf_orient = []
    # panda_reset_close_gripper(pandaID)
    # # For drawer
    # for i in np.linspace(0, 0.35, 20):
    #     link_joint_index = link_index-1
    # # # For cupboard
    # # for i in np.linspace(-1.57, 0, 40)[::-1]:
    # #     link_joint_index = link_index-2
    #     pyb.resetJointState(kitchen, link_joint_index, i)
    #     shelf_traj.append(np.array(p.getLinkState(kitchen, link_index)[0]))
    #     shelf_orient.append(np.array(p.getLinkState(kitchen, link_index)[1]))
    # shelf_traj = np.array(shelf_traj)
    # shelf_orient = np.array(shelf_orient)
    # pyb.resetJointState(kitchen, link_joint_index, 0.0)
    # for _ in range(10):
    #     pyb.stepSimulation()
    # # choose a handle pose that the robot can reach.
    # handle_pose = np.array(pyb.getLinkState(kitchen, link_index)[0])
    # # add_debug_point(p, handle_pose[None, :])
    # # handle_orient = [0.0, np.pi/2, 0.0]
    # # handle_orient_q = pyb.getQuaternionFromEuler(handle_orient)
    # # For draw
    # R_H_T = np.array(
    #     [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    # )
    # # # For cabinet
    # # R_H_T = np.array([
    # #     [0, 0, -1], [0, -1, 0], [-1, 0, 0]
    # # ])
    # R_handle = Rot.from_quat(pyb.getLinkState(kitchen, link_index)[1])
    # R_target = R_handle.as_matrix()@R_H_T
    # handle_orient_q = Rot.from_matrix(R_target).as_quat()
    # # handle_pose = handle_pose + np.array([-0.05, 0.0, 0.0])
    # panda_reset_open_gripper(pandaID)
    # # # find the joint angles that will open the draw
    # # q_traj = []
    # # q_i = None
    # # for i, handle_pose in enumerate(shelf_traj):
    # #     R_handle = Rot.from_quat(shelf_orient[i])
    # #     R_target = R_handle.as_matrix()@R_H_T
    # #     handle_orient_q = Rot.from_matrix(R_target).as_quat()
    # #     q, _ = get_IK_pose(pandaID, jointsID, handle_pose, handle_orient_q, q_i)
    # #     # Check if robot is in collision w/ itself or kitchen.
    # #     while pu.get_distance(all_obstacles, pandaID)<0. or pu.check_self_collision(pandaID):
    # #         q, _ = get_IK_pose(pandaID, jointsID, handle_pose, handle_orient_q, q_i)
    # #     print(np.linalg.norm(np.array(pyb.getLinkState(pandaID, 11)[0])-handle_pose))
    # #     q_i = q
    # #     q_traj.append(np.array(q)[:7])
    
    # with open(f'handle_{link_index}_traj.pkl', 'wb') as f:
    #     pickle.dump({'traj':q_traj}, f)
    # pyb.resetJointState(kitchen, link_joint_index, 0.35)
    # with open(f'handle_{link_index}_traj.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # Open trajectory
    # q_traj = data['traj']
    # # Close trajectory
    # q_traj = data['traj'][::-1]
    # # execute the trajectory
    # # # ------------- Move the robot ------------------------------------
    # # Place the arm at the handle
    # pu.set_position(pandaID, jointsID, q_traj[0])
    # for _ in range(100):
    #     panda_close_gripper(p, pandaID)
    #     p.stepSimulation()
    
    # for q_i in q_traj[:]:
    #     # pu.set_position(pandaID, jointsID, q_i)
    #     # for j in range(50):            
    #     #     # # Close the grips
    #     j_c = np.array(list(
    #         map(lambda x: x[0], p.getJointStates(pandaID, jointsID))
    #     ))
    #     count = 0
    #     while np.linalg.norm(j_c-q_i)>1e-2 and count<10:
    #         j_c = np.array(list(
    #         map(lambda x: x[0], p.getJointStates(pandaID, jointsID))
    #         ))
    #         p.setJointMotorControlArray( 
    #             pandaID,
    #             jointsID,
    #             pyb.POSITION_CONTROL,
    #             targetPositions=q_i
    #         )
    #         p.stepSimulation()
    #         time.sleep(0.1)
    #         count += 1