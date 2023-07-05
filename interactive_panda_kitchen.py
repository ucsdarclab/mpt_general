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

import torch
from torch.distributions import MultivariateNormal
import torch_geometric.data as tg_data

import open3d as o3d
import json

from ompl_utils import get_ompl_state, get_numpy_state

# VQ-MPT model 
from modules.quantizers import VectorQuantizer
from modules.decoder import DecoderPreNormGeneral
from modules.encoder import EncoderPreNorm
from modules.autoregressive import AutoRegressiveModel, EnvContextCrossAttModel

import eval_const_7d as ec7

def add_debug_point(client_id, pose):
    colors = np.zeros_like(pose)
    colors[:, 0] = 1.0
    obj_id = client_id.addUserDebugPoints(pose, colors, 25)
    return obj_id

def panda_reset_open_gripper(client_id, robotID, gripper_dist=0.02):
    '''
    Open the grippers for the panda robot.
    :param robotID: pybullet robot id for the panda arm.
    :param gripper_dis: distance between the gripper.
    '''
    client_id.resetJointState(robotID, 9, gripper_dist/2)
    client_id.resetJointState(robotID, 10, gripper_dist/2)

def panda_reset_close_gripper(client_id, robotID):
    '''
    Close the grippers for the panda robot.
    :param robotID: pybullet robot id for the panda arm.
    '''
    client_id.resetJointState(robotID, 9, 0.0)
    client_id.resetJointState(robotID, 10, 0.0)

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
        targetVelocities=-np.ones(2)*0.1
    )

def panda_open_gripper(clientID, robotID, gripper_dist=0.01):
    '''
    Close the gripper using velocity control.
    :param clientID: pybullet client id.
    :param robotID: pybullet robot id for the panda arm.
    '''
    clientID.setJointMotorControlArray(
        robotID,
        [9, 10],
        pyb.POSITION_CONTROL,
        targetPositions=np.ones(2)*gripper_dist
    )

def panda_stop_gripper(clientID, robotID):
    '''
    Send zero velocity to the gripper.
    :param clientID: bullet client id.
    :param robotID: pybullet robot id.
    '''
    clientID.setJointMotorControlArray(
        robotID,
        [9, 10],
        pyb.VELOCITY_CONTROL,
        targetVelocities=np.zeros(2)
    )

def get_IK_pose(client_id, robotID, jointsID, ee_pose, ee_orient, init_q=None, link_id=11):
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
    pu.set_position(client_id, robotID, jointsID, q)
    attempt = 0
    while np.linalg.norm(np.array(client_id.getLinkState(robotID, link_id)[0])-ee_pose)>1e-4:
        q = client_id.calculateInverseKinematics(
            robotID,
            link_id,
            ee_pose,
            ee_orient,
            maxNumIterations=75
        )
        pu.set_position(client_id, robotID, jointsID, q)
        attempt +=1
        if attempt>5:
            return q, False
    return q, True

from tracikpy import TracIKSolver

def get_IK_posev2(client_id, robotID, jointsID, eeT, q_init=None):
    '''
    Find the robot joint values for the given end-effector pose using
    Options for IK solver:
    Speed: returns very quickly the first solution found
    Distance: runs for the full timeout_in_secs, then returns the solution that minimizes SSE from the seed
    Manipulation1: runs for full timeout, returns solution that maximizes sqrt(det(J*J^T)) (the product of the singular values of the Jacobian)
    Manipulation2: runs for full timeout, returns solution that minimizes the ratio of min to max singular values of the Jacobian.
    TracIKSolver.
    :param robotID: 
    :param jointsID:
    :param eeT:
    '''
    ik_solver = TracIKSolver(
        "assets/franka_panda/franka_panda.urdf",
        "panda_link0",
        "panda_hand",
        # "panda_rightfinger",
        timeout=0.1,
        solve_type='Distance'
    )
    qout = ik_solver.ik(eeT, q_init)
    if qout is None:
        qout = (pu.q_min+np.random.rand(7)*(pu.q_max-pu.q_min))[0]
        return qout, False
    pu.set_position(client_id, robotID, jointsID, qout)
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

        while True:
            yield random_samples[index, :]
            index += 1
            if index==self.seq_num:
                random_samples = np.random.permutation(self.X.sample()*(self.q_max-self.q_min)+self.q_min)
                index = 0
                
    def sampleUniform(self, state):
        '''Generate a sample from uniform distribution or key-points
        :param state: ompl.base.Space object
        '''
        if self.X is None:
            sample_pos = ((self.q_max-self.q_min)*self.U.rvs()+self.q_min)[0]
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


def get_camera_matrix(camTheta, camPhi):
    '''
    Randomly sample camera positions from different angles, and positions
    :param camTheat: yaw angle of camera.
    :param camPhi: pitch angle of camera.
    :returns (np.array, np.array, np.array): Returns the view and projection Matrix.
    '''
    # Randomly sample camera positions
    camR = 2
    cameraEyePosition = np.r_[(
        camR*np.cos(camPhi)*np.cos(camTheta),
        camR*np.cos(camPhi)*np.sin(camTheta),
        camR*np.sin(camPhi)
    )]
    # The camera Up Vector is perpendicular to the EyePosition vector
    # and tangent to the sphere.
    cameraUpVector = np.r_[(
        -cameraEyePosition[0],
        -cameraEyePosition[1],
        (cameraEyePosition[0]**2+cameraEyePosition[1]**2)/cameraEyePosition[2]
    )] 
    cameraUpVector = cameraUpVector/np.linalg.norm(cameraUpVector)

    viewMatrix = pyb.computeViewMatrix(
        cameraEyePosition=cameraEyePosition,
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=cameraUpVector
    )

    # Projection Matrix
    projectionMatrix = pyb.computeProjectionMatrixFOV(
        fov=45.0,
        aspect=1.0,
        nearVal=1,
        farVal=5.1
    )
    return viewMatrix, projectionMatrix, cameraEyePosition

def getCameraDepthImage(client_id, camTheta, camPhi):
    '''
    return the depth image by having the camera for the desired pose.
    :param client_id: pybullet client id
    :param camTheta: yaw angle of camera
    :param camPhi: pitch angle of camera.
    :returns (np.array, int, int, np.array): depth image, height, width, world_to_pix transformation.
    '''
    viewMatrix, projMatrix,  _ = get_camera_matrix(camTheta, camPhi)
    width, height, rgbImg, depthImg, _ = client_id.getCameraImage(width=224, height=224, viewMatrix=viewMatrix, projectionMatrix=projMatrix)
    vM = np.reshape(viewMatrix, (4, 4), order='F')
    pM = np.reshape(projMatrix, (4, 4), order='F')
    tran_pix_world = np.linalg.inv(pM@vM)
    return np.array(depthImg), np.array(rgbImg), height, width, tran_pix_world


# bounding_box = o3d.geometry.AxisAlignedBoundingBox(np.ones(3)*-1.2, np.ones(3)*1.2)
bounding_box = o3d.geometry.AxisAlignedBoundingBox([-0.88, -0.88, -.4], [0.88, 0.88, 1.2])
def get_pcd(client_id):
    '''
    Using virtual cameras generate a PCD
    :param client_id: pybullet client ID.
    :return open3d.geometry.PointCloud: open3d pc object.
    '''
    pcdCollection = []
    for phi in [40*np.pi/180, -40*np.pi/180]:
        # for theta in [np.pi/4,-np.pi/4, 3*np.pi/4, -3*np.pi/4]:
        for theta in [3*np.pi/4, -3*np.pi/4]:
            depthImg, colorImg, height, width, tran_pix_world = getCameraDepthImage(client_id, theta, phi)
            pcd = depth2pc_opengl(depthImg, tran_pix_world, height, width)
            pcdCollection.append(pcd)
    pcdAllPoints = o3d.geometry.PointCloud()
    pcdAllPoints.points = o3d.utility.Vector3dVector(np.concatenate(pcdCollection))
    # Cropped point cloud, NOTE: This can be reduced further!!
    cropped_pcd = pcdAllPoints.crop(bounding_box)
    # return cropped_pcd.random_down_sample(0.4)
    return cropped_pcd.random_down_sample(0.7)

def follow_trajectory(client_id, robotID, jointsID, q_traj):
    '''
    :param client_id: bullet client object.
    :param robotID: pybullet id for robot.
    :param jointsID: pybullet robot links to control
    :param q_traj: The trajectory to follow:
    '''
    for q_i in q_traj[:]:
        # Get the current joint state
        j_c = pu.get_joint_position(client_id, robotID, jointsID)
        count = 0
        # Apply control till robot reaches final goal, or after 10
        # position updates
        while np.linalg.norm(j_c-q_i)>1e-2 and count<10:
            client_id.setJointMotorControlArray(
                robotID,
                jointsID,
                pyb.POSITION_CONTROL,
                targetPositions=q_i
            )
            client_id.stepSimulation()
            time.sleep(0.01)
            # Update current position
            j_c = pu.get_joint_position(client_id, robotID, jointsID)
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
    # NOTE: Currently hold-off on loading objects!!
    # Start postion of the object
    # itm_id = client_id.loadURDF(path_to_urdf, [0.555, -0.385, 0.642])
    itm_id = -1
    return all_obstacles, itm_id

from time import perf_counter

class Timer:
    '''
    From https://realpython.com/python-with-statement/#performing-high-precision-calculations
    '''
    def __enter__(self):
        self.start = perf_counter()
        self.end = 0.0
        return lambda: self.end - self.start

    def __exit__(self, *args):
        self.end = perf_counter()

if __name__=="__main__":
    seed = 100
    use_model = True
    # Server for collision checking
    p_collision = pu.get_pybullet_server('direct')
    # Server for collision checking
    p_pcd = pu.get_pybullet_server('direct')
    # Server for visualization/execution
    # p = pu.get_pybullet_server('gui')
    p = pu.get_pybullet_server('direct')

    p.resetDebugVisualizerCamera(1.8, -28.6, -33.6, np.array([0.0, 0.0, 0.0]))
    # p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setAdditionalSearchPath(osp.join(os.getcwd(), 'assets'))

        # ============== Load VQ-MPT Model ======================
    dict_model_folder = '/root/data/general_mpt_panda_7d/model1'
    ar_model_folder = '/root/data/general_mpt_panda_7d/stage2/model1'
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    # Define the models
    d_model = 512
    #TODO: Get the number of keys from the saved data
    num_keys = 2048
    goal_index = num_keys + 1
    quantizer_model = VectorQuantizer(n_e=num_keys, e_dim=8, latent_dim=d_model)

    # Load quantizer model.
    dictionary_model_folder = dict_model_folder
    with open(osp.join(dictionary_model_folder, 'model_params.json'), 'r') as f:
        dictionary_model_params = json.load(f)

    encoder_model = EncoderPreNorm(**dictionary_model_params)
    decoder_model = DecoderPreNormGeneral(
        e_dim=dictionary_model_params['d_model'], 
        h_dim=dictionary_model_params['d_inner'], 
        c_space_dim=dictionary_model_params['c_space_dim']
    )

    checkpoint = torch.load(osp.join(dictionary_model_folder, 'best_model.pkl'))
    
    # Load model parameters and set it to eval
    for model, state_dict in zip([encoder_model, quantizer_model, decoder_model], ['encoder_state', 'quantizer_state', 'decoder_state']):
        model.load_state_dict(checkpoint[state_dict])
        model.eval()
        model.to(device)

    # Load the AR model.
    # NOTE: Save these values as dictionary in the future, and load as json.
    env_params = {
        'd_model': dictionary_model_params['d_model'],
    }
    # Create the environment encoder object.
    with open(osp.join(ar_model_folder, 'cross_attn.json'), 'r') as f:
        context_env_encoder_params = json.load(f)
    context_env_encoder = EnvContextCrossAttModel(env_params, context_env_encoder_params, robot='6D')
    # Create the AR model
    with open(osp.join(ar_model_folder, 'ar_params.json'), 'r') as f:
        ar_params = json.load(f)
    ar_model = AutoRegressiveModel(**ar_params)

    # Load the parameters and set the model to eval
    checkpoint = torch.load(osp.join(ar_model_folder, 'best_model.pkl'))
    for model, state_dict in zip([context_env_encoder, ar_model], ['context_state', 'ar_model_state']):
        model.load_state_dict(checkpoint[state_dict])
        model.eval()
        model.to(device)
    run_data = []
    for _ in range(5):
        print("Resetting Simulation")
        for client_id in [p, p_pcd, p_collision]:
            client_id.resetSimulation()
        
        timing_dict = {}
        # Set up environment for simulation
        all_obstacles, itm_id = set_env(p)
        kitchen = all_obstacles[0]
        # Set up environment for collision checking
        all_obstacles_coll, itm_id_coll = set_env(p_collision)
        # Set up environment for capturing pcd
        all_obstacles_pcd, itm_id_pcd = set_env(p_pcd)

        # Load the interactive robot
        pandaID, jointsID, fingerID = pu.set_robot(p)
        # Load the collision checking robot
        pandaID_col, jointsID_col, fingerID_col = pu.set_robot(p_collision)
        
        # ============== Get PCD ===========
        pcd = get_pcd(p_pcd)

        # ============== Trajectory planning =================
        # Define OMPL parameters
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

        np.random.seed(seed)
        # Randomly sample a collision free start point.
        initial_config = (pu.q_min + (pu.q_max-pu.q_min)*np.random.rand(7))[0]
        pu.set_position(p_collision, pandaID_col, jointsID_col, initial_config)
        while pu.get_distance(p_collision, all_obstacles_coll, pandaID_col)<0. or pu.check_self_collision(p_collision, pandaID_col):
            initial_config = (pu.q_min + (pu.q_max-pu.q_min)*np.random.rand(7))[0]
            pu.set_position(p_collision, pandaID_col, jointsID_col, initial_config)
        pu.set_position(p, pandaID, jointsID, initial_config)
        # Plan a trajectory from initial config to cupboard handle grasp location.
        goal_q = q_traj[0]
        if use_model:
            n_start_n_goal = (np.r_[initial_config[None, :], goal_q[None, :]]-pu.q_min)/(pu.q_max-pu.q_min)
            map_data = tg_data.Data(pos=torch.as_tensor(np.asarray(pcd.points), dtype=torch.float, device=device))
            search_dist_mu, search_dist_sigma, _ = ec7.get_search_dist(
                n_start_n_goal, 
                None, 
                map_data, 
                context_env_encoder, 
                decoder_model, 
                ar_model, 
                quantizer_model, 
                num_keys
            )
        else:
            search_dist_mu, search_dist_sigma = None, None
        with Timer() as timer:
            traj_cupboard, _, _ , success = get_path(initial_config, goal_q, validity_checker_obj, search_dist_mu, search_dist_sigma)
            follow_trajectory(p, pandaID, jointsID, traj_cupboard)
        timing_dict['cupboard_handle_reach'] = timer()

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
        panda_reset_open_gripper(p_collision, pandaID_col, gripper_dist=2*gripper_joint_state)

        # Sync collision and pcd env
        cupboard_joint_state = p.getJointState(kitchen, door_link_index-2)[0]
        p_collision.resetJointState(all_obstacles_coll[0], door_link_index-2, cupboard_joint_state)
        p_pcd.resetJointState(all_obstacles_pcd[0], door_link_index-2, cupboard_joint_state)

        # Sync gripper.
        panda_reset_open_gripper(p_collision, pandaID_col, 0.1)

        with open('shelf_reach_q.pkl', 'rb') as f:
            data = pickle.load(f)
            can_start_q = data['start_q']
        with open('shelf_target_q.pkl', 'rb') as f:
            data = pickle.load(f)
            can_goal_q = data['goal_q']

        tmp_start_q = pu.get_joint_position(p, pandaID, jointsID)
        
        if use_model:
            pcd = get_pcd(p_pcd)
            n_start_n_goal = (np.r_[tmp_start_q[None, :], can_start_q[None, :]]-pu.q_min)/(pu.q_max-pu.q_min)
            map_data = tg_data.Data(pos=torch.as_tensor(np.asarray(pcd.points), dtype=torch.float, device=device))
            search_dist_mu, search_dist_sigma, _ = ec7.get_search_dist(
                n_start_n_goal, 
                None, 
                map_data, 
                context_env_encoder, 
                decoder_model, 
                ar_model, 
                quantizer_model, 
                num_keys
            )
        else:
            search_dist_mu, search_dist_sigma = None, None

        with Timer() as timer:
            path_cupboard_2_can, _, _, success = get_path(tmp_start_q, can_start_q, validity_checker_obj, search_dist_mu, search_dist_sigma)
            # Execute cupboard trajectory
            follow_trajectory(p, pandaID, jointsID, path_cupboard_2_can)
        timing_dict['can_reach'] = timer()

        for _ in range(10):
            p.stepSimulation()
        # Close the gripper
        for _ in range(200):
            panda_close_gripper(p, pandaID)
            p.stepSimulation()
        # Plan a trajectory from grasp point to table
        if use_model:
            pcd = get_pcd(p_pcd)
            n_start_n_goal = (np.r_[can_start_q[None, :], can_goal_q[None, :]]-pu.q_min)/(pu.q_max-pu.q_min)
            map_data = tg_data.Data(pos=torch.as_tensor(np.asarray(pcd.points), dtype=torch.float, device=device))
            search_dist_mu, search_dist_sigma, _ = ec7.get_search_dist(
                n_start_n_goal, 
                None, 
                map_data, 
                context_env_encoder, 
                decoder_model, 
                ar_model, 
                quantizer_model, 
                num_keys
            )
        else:
            search_dist_mu, search_dist_sigma = None, None

        with Timer() as timer:
            path_can, _, _, success = get_path(can_start_q, can_goal_q, validity_checker_obj, search_dist_mu, search_dist_sigma)
            follow_trajectory(p, pandaID, jointsID, path_can)
        timing_dict['table_reach'] = timer()

        run_data.append(timing_dict)
    
    if use_model:
        file_name = f'kitchen_timing_data_vqmpt_{seed}.pkl'
    else:
        file_name = f'kitchen_timing_data_{seed}.pkl'
    
    with open(file_name, 'wb') as f:
        pickle.dump(run_data, f)