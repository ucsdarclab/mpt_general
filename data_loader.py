'''Data loader for the 2D planner.
'''

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import skimage.io

import pickle
import re
import numpy as np
import open3d as o3d

import os
from os import path as osp

import torch_geometric.data as tg_data

from panda_utils import q_max, q_min

class PathManipulationDataLoader(Dataset):
    ''' Loads each path for the maniuplation data.
    '''

    def __init__(self, data_folder, env_list):
        '''
        :param data_folder: location of where file exists. 
        '''
        self.data_folder = data_folder
        self.index_dict = [(envNum, int(re.findall('[0-9]+', filei)[0]))
                           for envNum in env_list
                           for filei in os.listdir(osp.join(data_folder, f'env_{envNum:06d}'))
                           if filei.endswith('.p')
                           ]

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.index_dict)

    def __getitem__(self, index):
        '''Gets the data item from a particular index.
        :param index: Index from which to extract the data.
        :returns: A dictionary with path.
        '''
        envNum, pathNum = self.index_dict[index]
        envFolder = osp.join(self.data_folder, f'env_{envNum:06d}')

        #  Load the path
        with open(osp.join(envFolder, f'path_{pathNum}.p'), 'rb') as f:
            data_path = pickle.load(f)
            joint_path = data_path['jointPath']
        # Normalize the trajectory.
        q = (joint_path-q_min)/(q_max-q_min)
        return {'path': torch.as_tensor(q[:, :6])}


def get_quant_manipulation_sequence(batch):
    '''
    This should be passed to DataLoader class to collate batched samples with various lengths.
    :param batch: the batch to consolidate
    '''
    data = {}
    data['map'] = tg_data.Batch.from_data_list(
        [batch_i['map'] for batch_i in batch])
    data['input_seq'] = pad_sequence(
        [batch_i['input_seq'] for batch_i in batch], batch_first=True)
    data['target_seq_id'] = pad_sequence([batch_i['target_seq_id']
                                          for batch_i in batch], batch_first=True)
    data['length'] = torch.tensor(
        [batch_i['input_seq'].shape[0]+1 for batch_i in batch])
    data['start_n_goal'] = torch.cat(
        [batch_i['start_n_goal'][None, :] for batch_i in batch])
    return data


class QuantManipulationDataLoader(Dataset):
    ''' Data loader for quantized data values and associated point cloud.
    '''

    def __init__(self,
                 quantizer_model,
                 env_list,
                 map_data_folder,
                 quant_data_folder):
        '''
        :param quantizer_model: The quantizer model to use.
        :param env_list: List of environments to use for training.
        :param map_data_folder: location of the point cloud data.
        :param quant_data_folder: location of quantized data folder
        '''
        self.quant_data_folder = quant_data_folder
        self.map_data_folder = map_data_folder
        self.index_dict = [(envNum, int(re.findall('[0-9]+', filei)[0]))
                           for envNum in env_list
                           for filei in os.listdir(osp.join(quant_data_folder, f'env_{envNum:06d}'))
                           if filei.endswith('.p')
                           ]
        self.quantizer_model = quantizer_model

        total_num_embedding = quantizer_model.embedding.weight.shape[0]
        self.start_index = total_num_embedding
        self.goal_index = total_num_embedding + 1

    def __len__(self):
        ''' Return the length of the dataset.
        '''
        return len(self.index_dict)

    def __getitem__(self, index):
        ''' Return the PC of the env and quant data.
        :param index: The index of the data.
        '''
        env_num, path_num = self.index_dict[index]

        # Load the pcd data.
        data_folder = osp.join(self.map_data_folder, f'env_{env_num:06d}')
        map_file = osp.join(data_folder, f'map_{env_num}.ply')
        data_PC = o3d.io.read_point_cloud(map_file, format='ply')
        depth_points = np.array(data_PC.points)

        # Load start and goal states.
        with open(osp.join(data_folder, f'path_{path_num}.p'), 'rb') as f:
            data_path = pickle.load(f)
            joint_path = data_path['jointPath']
            # Normalize the trajectory.
            start_n_goal = ((joint_path-q_min)/(q_max-q_min))[[0, -1], :6]

        # Load the quant-data
        with open(osp.join(self.quant_data_folder, f'env_{env_num:06d}', f'path_{path_num}.p'), 'rb') as f:
            quant_data = pickle.load(f)

        with torch.no_grad():
            quant_vector = self.quantizer_model.embedding(
                torch.tensor(quant_data['keys']))
            quant_proj_vector = self.quantizer_model.output_linear_map(
                quant_vector)

        # add start vector and goal vector:
        input_seq = torch.cat(
            [torch.ones(1, 512)*-1, quant_proj_vector, torch.ones(1, 512)], dim=0)
        input_seq_keys = np.r_[self.start_index,
                               quant_data['keys'], self.goal_index]

        return {
            'map': tg_data.Data(pos=torch.as_tensor(depth_points, dtype=torch.float)),
            'start_n_goal': torch.as_tensor(start_n_goal, dtype=torch.float),
            'input_seq': input_seq[:-1],
            'target_seq_id': torch.as_tensor(input_seq_keys[1:])
        }


def get_padded_sequence(batch):
    '''
    This should be passed to DataLoader class to collate batched samples with various length.
    :param batch: The batch to consolidate
    '''
    data = {}
    data['map'] = torch.cat([batch_i['map'][None, :]
                            for batch_i in batch if batch_i is not None])
    data['path'] = pad_sequence(
        [batch_i['path'] for batch_i in batch if batch_i is not None], batch_first=True)
    data['mask'] = pad_sequence([torch.ones(batch_i['path'].shape[0])
                                for batch_i in batch if batch_i is not None], batch_first=True)
    return data


class PathMixedDataLoader(Dataset):
    '''Loads each path, and extracts the masked positive and negative regions.
    The data is indexed in such a way that "hard" planning problems are equally distributed
    uniformly throughout the dataloading process.
    '''

    def __init__(self, envListMaze, dataFolderMaze, envListForest, dataFolderForest):
        '''
        :param envListMaze: The list of map environments to collect data from Maze.
        :param dataFolderMaze: The parent folder where the maze path files are located.
        :param envListForest: The list of map environments to collect data from Forest.
        :param dataFodlerForest: The parent folder where the forest path files are located.
            It should follow the following format:
                env1/path_0.p
                    ...
                env2/path_0.p
                    ...
                    ...
        '''
        assert isinstance(envListMaze, list), "Needs to be a list"
        assert isinstance(envListForest, list), "Needs to be a list"

        self.num_env = len(envListForest) + len(envListMaze)
        self.indexDictMaze = [('M', envNum, i)
                              for envNum in envListMaze
                              for i in range(len(os.listdir(osp.join(dataFolderMaze, f'env{envNum:06d}')))-1)
                              ]
        self.indexDictForest = [('F', envNum, i)
                                for envNum in envListForest
                                for i in range(len(os.listdir(osp.join(dataFolderForest, f'env{envNum:06d}')))-1)
                                ]
        self.dataFolder = {'F': dataFolderForest, 'M': dataFolderMaze}
        self.envList = {'F': envListForest, 'M': envListMaze}

    def __len__(self):
        return len(self.indexDictForest)+len(self.indexDictMaze)

    def __getitem__(self, idx):
        '''
        Returns the sample at index idx.
        returns dict: A dictonary of the encoded map and target points.
        '''
        DF, env, idx_sample = idx
        dataFolder = self.dataFolder[DF]
        mapEnvg = skimage.io.imread(
            osp.join(dataFolder, f'env{env:06d}', f'map_{env}.png'), as_gray=True)

        with open(osp.join(dataFolder, f'env{env:06d}', f'path_{idx_sample}.p'), 'rb') as f:
            data = pickle.load(f)

        if data['success']:
            path = data['path_interpolated']/24
            # path = data['path']/24
            return {
                'map': torch.as_tensor(mapEnvg),
                'path': torch.as_tensor(path)
            }


def get_quant_padded_sequence(batch):
    '''
    This should be passed to DataLoader class to collate batched samples with various length.
    :param batch: The batch to consolidate
    '''
    data = {}
    data['map'] = torch.cat([batch_i['map'][None, :]
                            for batch_i in batch])
    data['input_seq'] = pad_sequence(
        [batch_i['input_seq'] for batch_i in batch], batch_first=True)
    data['target_seq_id'] = pad_sequence([batch_i['target_seq_id']
                                for batch_i in batch], batch_first=True)
    data['length'] = torch.tensor([batch_i['input_seq'].shape[0]+1 for batch_i in batch])
    data['start_n_goal'] = torch.cat([batch_i['start_n_goal'][None, :] for batch_i in batch])
    return data

class QuantPathMixedDataLoader(Dataset):
    '''Loads the qunatized path.
    '''

    def __init__(
            self, 
            quantizer_model, 
            envListMaze,
            dataFolderMaze,
            quant_data_folder_maze,
            envListForest, 
            dataFolderForest,
            quant_data_folder_forest
        ):
        '''
        :param envListMaze: The list of map environments to collect data from Maze.
        :param dataFolderMaze: The parent folder where the maze path files are located.
        :param envListForest: The list of map environments to collect data from Forest.
        :param dataFodlerForest: The parent folder where the forest path files are located.
            It should follow the following format:
                env1/path_0.p
                    ...
                env2/path_0.p
                    ...
                    ...
        '''
        assert isinstance(envListMaze, list), "Needs to be a list"
        assert isinstance(envListForest, list), "Needs to be a list"

        self.num_env = len(envListForest) + len(envListMaze)
        self.indexDictMaze = [('M', envNum, int(re.findall(r'\d+', f)[0]))
                              for envNum in envListMaze
                              for f in os.listdir(osp.join(quant_data_folder_maze, f'env{envNum:06d}')) if f[-2:]=='.p'
                              ]
        self.indexDictForest = [('F', envNum, int(re.findall(r'\d+', f)[0]))
                                for envNum in envListForest
                                for f in os.listdir(osp.join(quant_data_folder_forest, f'env{envNum:06d}')) if f[-2:]=='.p'
                                ]
        self.dataFolder = {'F': dataFolderForest, 'M': dataFolderMaze}
        self.quant_data_folder = {'F': quant_data_folder_forest, 'M': quant_data_folder_maze}
        self.quantizer_model = quantizer_model
        
        total_num_embedding = quantizer_model.embedding.weight.shape[0]
        self.start_index = total_num_embedding
        self.goal_index = total_num_embedding + 1

    def __len__(self):
        return len(self.indexDictForest)+len(self.indexDictMaze)

    def __getitem__(self, idx):
        '''
        Returns the sample at index idx.
        returns dict: A dictonary of the encoded map and target points.
        '''
        DF, env, idx_sample = idx
        dataFolder = self.dataFolder[DF]
        quant_data_folder = self.quant_data_folder[DF]
        
        map_env = skimage.io.imread(
            osp.join(dataFolder, f'env{env:06d}', f'map_{env}.png'), as_gray=True)

        with open(osp.join(quant_data_folder, f'env{env:06d}', f'path_{idx_sample}.p'), 'rb') as f:
            quant_data = pickle.load(f)
            
        with open(osp.join(dataFolder, f'env{env:06d}', f'path_{idx_sample}.p'), 'rb') as f:
            data = pickle.load(f)
    
        with torch.no_grad():
            quant_vector = self.quantizer_model.embedding(torch.tensor(quant_data['keys']))
            quant_proj_vector = self.quantizer_model.output_linear_map(quant_vector)

        # add start vector and goal vector:
        input_seq = torch.cat([torch.ones(1, 512)*-1, quant_proj_vector, torch.ones(1, 512)], dim=0)
        input_seq_keys = np.r_[self.start_index, quant_data['keys'], self.goal_index]
        # Normalize the start and goal points
        start_n_goal = data['path'][[0,-1], :]/24
        return {
            'map': torch.as_tensor(map_env[None, :], dtype=torch.float),
            'start_n_goal': torch.as_tensor(start_n_goal, dtype=torch.float),
            'input_seq': input_seq[:-1],
            'target_seq_id': torch.as_tensor(input_seq_keys[1:])
        }