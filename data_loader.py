'''Data loader for the 2D planner.
'''

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import skimage.io

import pickle

import numpy as np

import os
from os import path as osp


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
            return {
                'map': torch.as_tensor(mapEnvg),
                'path': torch.as_tensor(path)
            }
