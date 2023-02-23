''' Evaluate mpnet planner
'''

import argparse
from os import path as osp

import numpy as np
import pickle
import time
import open3d as o3d
import torch

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    raise "Run code from a container with OMPL installed"

from panda_utils import q_min, q_max
from mpnet_models import MLP, Encoder
from ompl_utils import get_numpy_state, get_ompl_state
from panda_utils import set_env, get_pybullet_server


def scale_state(state):
    '''
    Scales trajectory from [-1, 1] back to q_min:q_max
    '''
    scaled_state = np.zeros((state.shape[0], 7))
    scaled_state[:, :6] = state
    return (q_max-q_min)*(scaled_state+1)/2 + q_min


def construct_traj(start_pose, goal_pose):
    '''
    Construct a linear trajectory from the start to goal pose.
    :param start_pose:
    :param goal_pose:
    :returns np.array: A npy array of states.
    '''
    alpha = np.arange(0, 1, step=0.01)[:, None]
    return start_pose*(1-alpha)+goal_pose*alpha

def valid_local_traj(traj, space, validity_checker):
    ''' Test the local trajectory.
    :param traj: np.array of trajectory
    :return bool: Returns True, if local traj is collision free.
    '''
    for p in traj:
        test_state = get_ompl_state(space, p)
        if not validity_checker.isValid(test_state):
            return False
    return True

def get_predict_points(start_tensor, goal_tensor, h, mlp_model, space, validity_checker, max_pred_steps):
    ''' Generate points connecting the start and goal tensor.
    '''
    is_connected = False
    cur_state = start_tensor[None, :]
    end_state = goal_tensor[None, :]
    forward_pred = [scale_state(cur_state.numpy())]
    reverse_pred = [scale_state(end_state.numpy())]
    forward = True
    for _ in range(max_pred_steps):
        if forward:
            mlp_input = torch.cat((cur_state, end_state, h), dim=1)
            next_state = mlp_model(mlp_input)
            while not validity_checker.isValid(scale_state(next_state.detach().numpy()).squeeze()):
                next_state = mlp_model(mlp_input)
            # TODO: Check if we can connect the predict state with end state
            scaled_state = scale_state(next_state.detach().numpy())
            local_traj = construct_traj(scaled_state, reverse_pred[0])
            # If valid path exists then return the points.
            if valid_local_traj(local_traj, space, validity_checker):
                is_connected = True
                forward_pred.append(scaled_state)
                print("Connected")
                break
            forward_pred.append(scaled_state)
            cur_state = next_state
            forward=False
        else:
            mlp_input = torch.cat((end_state, cur_state, h), dim=1)
            next_state = mlp_model(mlp_input)
            while not validity_checker.isValid(scale_state(next_state.detach().numpy()).squeeze()):
                next_state = mlp_model(mlp_input)
            scaled_state = scale_state(next_state.detach().numpy())
            local_traj = construct_traj(forward_pred[-1], scaled_state)
            if valid_local_traj(local_traj, space, validity_checker):
                is_connected = True
                reverse_pred.insert(0, scaled_state)
                print("Connected")
                break
            reverse_pred.insert(0, scaled_state)
            end_state = next_state
            forward=True
    return np.array(forward_pred+reverse_pred).squeeze(), is_connected


# If not able to connect segments, Run RRT.
def get_path_segment(start, goal, space, validity_checker_obj, plan_time=2):
    '''
    Plan ur path segment
    '''
    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(validity_checker_obj)

    start_state = get_ompl_state(space, start)
    goal_state = get_ompl_state(space, goal)

    success = False

    # Define planning problem
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_state, goal_state)

    planner = og.RRT(si)
    planner_type = 'rrt'

    planner.setProblemDefinition(pdef)
    planner.setup()

    # Attempt to solve the planning problem in the given time
    solved = planner.solve(plan_time)
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
        # TODO: Get final planner path. 
        path = [
            get_numpy_state(pdef.getSolutionPath().getState(i))
            for i in range(pdef.getSolutionPath().getStateCount())
            ]
    else:
        path = [start, goal]

    return path, success


def get_mpnet_path(q, mlp_model, h, p, env_num):
    '''
    '''
    success = False
    # Planning parameters
    space = ob.RealVectorStateSpace(7)
    bounds = ob.RealVectorBounds(7)
    # Set joint limits
    for i in range(7):
        bounds.setHigh(i, q_max[0, i])
        bounds.setLow(i, q_min[0, i])
    space.setBounds(bounds)

    validity_checker_obj = set_env(p, space, 6, 6, seed=env_num)

    # Try connecting the ends
    for _ in range(10):
        pred_traj, success = get_predict_points(q[0, :6], q[-1, :6], h, mlp_model, space, validity_checker_obj, 6)
    #     pred_f_traj, pred_r_traj, success = get_predict_points(q_tensor[0, :6], q_tensor[-1, :6], h, mlp_model, space, validity_checker_obj, 6)
        if success:
            break

    # If connected, check for path validity
    if success:
        final_traj = []
        for i, _ in enumerate(pred_traj[:-1]):
            local_traj = construct_traj(pred_traj[i], pred_traj[i+1])
            if not valid_local_traj(local_traj, space, validity_checker_obj):
                local_success = False
                print("Invalid Path")
                path_segment, local_success = get_path_segment(pred_traj[i], pred_traj[i+1], space, validity_checker_obj, 5)
                if not local_success:
                    success = False
                    print("No path found")
                    break
                else:
                    final_traj = final_traj + path_segment
            else:
                final_traj.append(pred_traj[i])
        final_traj.append(pred_traj[-1])
        return np.array(final_traj), success
    return None, success

def main(args):
    ''' pass arguments 
    '''
    p = get_pybullet_server('direct')

    enc_input_size=4914
    enc_output_size = 60
    mlp_output_size = 6
    max_num_points = args.max_num_points//3

    mlp_model = MLP(enc_output_size+mlp_output_size*2, mlp_output_size)
    encoder_model = Encoder(enc_input_size, enc_output_size)
    model_folder = args.model_folder

    epoch_num = 99
    checkpoint = torch.load(osp.join(model_folder, f'model_{epoch_num}.pkl'))
    mlp_model.load_state_dict(checkpoint['mlp_state'])
    encoder_model.load_state_dict(checkpoint['encoder_state'])

    data_folder = args.data_folder

    # ============================= Run planning experiment ============================
    start = args.start
    pathSuccess = []
    pathTime = []
    pathTimeOverhead = []
    pathVertices = []
    pathPlanned = []
    predict_seq_time = []
    

    path_num = 0
    for env_num in range(start, args.samples+start):
        env_folder = osp.join(data_folder, f'env_{env_num:06d}')
        with open(osp.join(env_folder, f'path_{path_num}.p'), 'rb') as f:
            data = pickle.load(f)
        if data['success']:
            q = 2*(data['jointPath']-q_min)/(q_max-q_min) - 1
            q = torch.tensor(q, dtype=torch.float)

            # Format point cloud data
            data_PC = o3d.io.read_point_cloud(osp.join(env_folder, f'map_{env_num}.ply'), format='ply')
            ratio = (max_num_points+2)/len(data_PC.points)
            downsamp_PC = data_PC.random_down_sample(ratio)
            depth_points = np.array(downsamp_PC.points)[:max_num_points]
            pc_data = torch.tensor(depth_points.reshape(-1)[None, :], dtype=torch.float)
            
            start_time = time.time()
            # Get encoder data
            h = encoder_model(pc_data)

            # Get path
            pred_traj, success = get_mpnet_path(q, mlp_model, h, p, env_num)
            plan_time = time.time()-start_time

            pathSuccess.append(success)
            pathTime.append(plan_time)
            if pred_traj is not None:
                pathPlanned.append(pred_traj)
                pathVertices.append(pred_traj.shape[0])
            else:
                pathPlanned.append(np.array(data['jointPath'][[0, -1]]))
                pathVertices.append(0.0)
            predict_seq_time.append(0.)
        else:
            pathSuccess.append(False)
            pathTime.append(0)
            pathVertices.append(0)
            pathTimeOverhead.append(0)
            pathPlanned.append([[]])
            predict_seq_time.append(0)

    pathData = {'Time':pathTime, 'Success':pathSuccess, 'Vertices':pathVertices, 'PlanTime':pathTimeOverhead, 'PredictTime': predict_seq_time, 'Path': pathPlanned}
    fileName = osp.join(model_folder, f'eval_val_plan_mpnet_{start:06d}.p')
    # fileName = osp.join(ar_model_folder, f'eval_val_plan_{args.planner_type}_shelf_{start:06d}.p')

    pickle.dump(pathData, open(fileName, 'wb'))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', help="Folder with MPNet model")
    parser.add_argument('--data_folder', help="Folder with valuation data")
    parser.add_argument('--max_num_points', help="Downsampling point cloud data", type=int)
    parser.add_argument('--start', help="Start environment", type=int)
    parser.add_argument('--samples', help="Number of samples to collect", type=int)

    args = parser.parse_args()
    main(args)