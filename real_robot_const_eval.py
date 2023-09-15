''' A script to test the planning model for panda
'''

from torch.nn import functional as F

import time
import skimage.io
from os import path as osp
from scipy import stats
from functools import partial
from torch.distributions import MultivariateNormal

import numpy as np
import torch
import json
import argparse
import pickle
import open3d as o3d
import torch_geometric.data as tg_data

import matplotlib.pyplot as plt

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    raise "Run code from a container with OMPL installed"

try:
    import rospy
    from moveit_msgs.srv import GetStateValidityRequest, GetStateValidity
    from moveit_msgs.msg import RobotState
    from sensor_msgs.msg import PointCloud2
except ImportError:
    raise "Make sure you have updated ROS_PACKAGE_PATH"

from modules.quantizers import VectorQuantizer
from modules.decoder import DecoderPreNorm, DecoderPreNormGeneral
from modules.encoder import EncoderPreNorm

from modules.autoregressive import AutoRegressiveModel, EnvContextCrossAttModel

import panda_utils as pu
import eval_const_7d as ec7
import interactive_kitchen_dev as ikd
import panda_constraint_shelf as pcs

from ompl_utils import get_ompl_state, get_numpy_state
import roboticstoolbox as rtb

res = 0.05
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

class PCStateValidityChecker(ob.StateValidityChecker):
    ''' State validity checker using point cloud data
    '''
    def __init__(self, si):
        super().__init__(si)
        # prepare service for collision check
        self.sv_srv = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
        # wait for service to become available
        self.sv_srv.wait_for_service()
        rospy.loginfo('service is available')
        # prepare msg to interface with moveit
        self.rs = RobotState()
        self.rs.joint_state.name = [
            'panda_joint1',
            'panda_joint2',
            'panda_joint3',
            'panda_joint4',
            'panda_joint5',
            'panda_joint6',
            'panda_joint7',
            'panda_finger_joint1',
            'panda_finger_joint2'
        ]
        self.rs.joint_state.position = [0]*len(self.rs.joint_state.name)
        self.rs.joint_state.position[7] = 0.040
        self.rs.joint_state.position[8] = 0.040
        self.joint_states_received = False

    def isValid(self, state):
        ''' Check if the given state is valid.
        :param state: ob.State object to be checked.
        :return bool: Ture if state is valid.
        '''
        if self.getStateValidity(state).valid:
            return True
        return False

    def getStateValidity(self, state, group_name='panda', constraints=None):
        '''
        Given a RobotState and a group name and an optional Constraints
        return the validity of the State
        '''
        gsvr = GetStateValidityRequest()
        for i in range(7):
            self.rs.joint_state.position[i]= state[i]
        gsvr.robot_state = self.rs

        gsvr.group_name = group_name
        if constraints != None:
            gsvr.constraints = constraints
        result = self.sv_srv.call(gsvr)
        return result

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


def getPathLengthObjective(cost, si):
    '''
    Return the threshold objective for early termination
    :param cost: The cost of the original RRT* path
    :param si: An object of class ob.SpaceInformation
    :returns : An object of class ob.PathLengthOptimizationObjective
    '''
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostThreshold(ob.Cost(cost))
    return obj
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_model_folder', help="Folder where dictionary model is stored")
    parser.add_argument('--ar_model_folder', help="Folder where AR model is stored")
    parser.add_argument('--samples', help="Number of samples to collect", type=int)
    parser.add_argument('--state_space', help="Types node expansion", choices=['PJ', 'AT', 'TB'])

    args = parser.parse_args()

    panda_model = rtb.models.DH.Panda()
    use_model = False if args.dict_model_folder is None else True
    if use_model:
        print("Using model")
        # ========================= Load trained model ===========================
        # Define the models
        d_model = 512
        num_keys = 2048
        goal_index = num_keys + 1
        quantizer_model = VectorQuantizer(n_e=num_keys, e_dim=8, latent_dim=d_model)

        # Load quantizer model.
        dictionary_model_folder = args.dict_model_folder
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
        ar_model_folder = args.ar_model_folder
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
    else:
        print("Comparing with vq-mpt planners")
        # Load VQ-MPT planned paths, for setting optimization objective.
        # with open(osp.join(args.ar_model_folder, f'eval_val_plan_rrt_{2000:06d}.p'), 'rb') as f:
        #     vq_mpt_data = pickle.load(f)
    # ============================= Run planning experiment ============================
    pathSuccess = []
    pathTime = []
    pathTimeOverhead = []
    pathVertices = []
    pathPlanned = []
    predict_seq_time = []
    
    path_try = 0

    # Planning parameters
    space = ob.RealVectorStateSpace(7)
    bounds = ob.RealVectorBounds(7)
    # Set joint limits
    for i in range(7):
        bounds.setHigh(i, pu.q_max[0, i])
        bounds.setLow(i, pu.q_min[0, i])
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)

    # Define a validity checker object
    validity_checker_obj = PCStateValidityChecker(si)

    # TODO: Define the start and goal pose!!
    # Config 1 
    config_num=5
    joint_path = np.array([
        [-2.4250823384067606, -1.2798519129000212, 1.2391232564993069, -2.0538458376625264, 0.26119280460145733, 2.953120983772808, 1.635956806363331],
        [0.06807870748063974, 1.6646113433837888, 1.0090846130425797, -2.452916619079385, -0.17679152705934312, 2.578336533241802, -1.2111141838395936]
    ])

    # TODO: Check the constraints 
    tolerance = np.array([2*np.pi, 0.1, 0.1])
    can_T_ee = np.array([[0., 0., 1, 0.], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    constraint_function = pcs.EndEffectorConstraint(can_T_ee[:3, :3], tolerance, None, None)
    
    # TODO: Define path_obj for the single path.
    path_obj = None
    if use_model:
        # TODO: Get point cloud from robot base!!
        # data_PC = o3d.io.read_point_cloud('map_2000.pcd')
        data_PC = o3d.io.read_point_cloud('crop_downsample_transform_scene.pcd')
        depth_points = np.array(data_PC.points)
        map_data = tg_data.Data(pos=torch.as_tensor(depth_points, dtype=torch.float, device=device))
        
        path = (joint_path-pu.q_min)/(pu.q_max-pu.q_min)
        # TODO: Get search Distribution
        search_dist_mu, search_dist_sigma, patch_time = None, None, None
    else:
        # Plan paths that are within 10% of the given path length.
        path_obj = None
        print("Not using model, using uniform distribution")
        search_dist_mu, search_dist_sigma, patch_time = None, None, 0.0

    # import pdb;pdb.set_trace()
    # Get a path for a given start and goal config
    planned_path, path_length, t, v , s = ikd.get_constraint_path_v2(
                    joint_path[0], 
                    joint_path[1], 
                    validity_checker_obj, 
                    constraint_function, 
                    search_dist_mu, 
                    search_dist_sigma,
                    plan_time=80
                )

    # Get path for a given TSR Region
    # planned_path, t, v, s = traj_cupboard, path_length, plan_time, num_vertices , success = ikd.get_constraint_path(
    #                         can_start_q, 
    #                         world_T_can, 
    #                         validity_checker_obj, 
    #                         constraint_function, 
    #                         search_dist_mu, 
    #                         search_dist_sigma,
    #                         plan_time=100,
    #                         state_space=args.state_space
    #                     )
    
    if s:
        print(f"Plan Time: {t}")
        if use_model:
            traj_file = f'traj_config_const_{config_num}.pkl'
        else:
            traj_file = f'traj_config_rrtconnect_const_{config_num}.pkl'
        with open(traj_file, 'wb') as f:
            pickle.dump({'robot_traj': planned_path},f)
    
    # Display planned path
    # robot = moveit_commander.RobotCommander()
    # group_name = "panda_arm"
    # move_group = moveit_commander.MoveGroupCommander(group_name)
    # display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    # display_trajectory.trajectory_start = robot.get_current_state()

    # pathSuccess.append(s)
    # pathTime.append(t)
    # pathVertices.append(v)
    # pathTimeOverhead.append(t)
    # pathPlanned.append(np.array(planned_path))
    # predict_seq_time.append(patch_time)


    # pathData = {'Time':pathTime, 'Success':pathSuccess, 'Vertices':pathVertices, 'PlanTime':pathTimeOverhead, 'PredictTime': predict_seq_time, 'Path': pathPlanned}
    # if use_model:
    #     fileName = osp.join(ar_model_folder, f'real_world_{args.state_space}_{0:06d}.p')
    # else:
    #     fileName = f'/root/data/general_mpt/real_world_{args.state_space}_{0:06d}.p'
    # pickle.dump(pathData, open(fileName, 'wb'))