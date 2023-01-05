''' Defining layers for converting maps to latent encodings.
'''

import torch
import torch.nn as nn

import numpy as np

from einops.layers.torch import Rearrange
from einops import rearrange

from modules.encoder import EncoderLayer


class PositionalEncoding(nn.Module):
    '''Positional encoding
    '''

    def __init__(self, d_hid, n_position):
        '''
        Intialize the Encoder.
        :param d_hid: Dimesion of the attention features.
        :param n_position: Number of positions to consider.
        :param train_shape: The 2D shape of the training model.
        '''
        super(PositionalEncoding, self).__init__()
        self.n_pos_sqrt = int(np.sqrt(n_position))
        # Not a parameter
        self.register_buffer('hashIndex', self._get_hash_table(n_position))
        self.register_buffer(
            'pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_hash_table(self, n_position):
        '''
        A simple table converting 1D indexes to 2D grid.
        :param n_position: The number of positions on the grid.
        '''
        return rearrange(torch.arange(n_position), '(h w) -> h w', h=int(np.sqrt(n_position)), w=int(np.sqrt(n_position)))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        '''
        Sinusoid position encoding table.
        :param n_position:
        :param d_hid:
        :returns 
        '''
        # TODO: make it with torch instead of numpy
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i)
                                  for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table[None, :])

    def forward(self, x, conv_shape=None):
        '''
        Callback function
        :param x:
        '''
        # assert x.shape[0]==1, "Only valid for testing single image sizes"
        selectIndex = rearrange(
            self.hashIndex[:conv_shape[0], :conv_shape[1]], 'h w -> (h w)')
        return x + torch.index_select(self.pos_table, dim=1, index=selectIndex)

# Encoder for environment
class EnvEncoder(nn.Module):
    ''' The environment encoder of the planner.
    '''

    def __init__(self, d_model, dropout, n_position):
        '''
        Intialize the encoder.
        :param n_layers: Number of layers of attention and fully connected layer.
        :param n_heads: Number of self attention modules.
        :param d_k: Dimension of each Key.
        :param d_v: Dimension of each Value.
        :param d_model: Dimension of input/output of encoder layer.
        :param d_inner: Dimension of the hidden layers of position wise FFN
        :param dropout: The value to the dropout argument.
        :param n_position: Total number of patches the model can handle.
        :param train_shape: The shape of the output of the patch encodings.
        '''
        super().__init__()
        # Convert the image to and input embedding.
        # NOTE: This is one place where we can add convolution networks.
        # Convert the image to linear model

        # NOTE: Padding of 3 is added to the final layer to ensure that
        # the output of the network has receptive field across the entire map.
        # NOTE: pytorch doesn't have a good way to ensure automatic padding. This
        # allows only for a select few map sizes to be solved using this method.
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16, d_model, kernel_size=5, stride=5, padding=3)
        )

        self.reorder_dims = Rearrange('b c h w -> b (h w) c')
        # Position Encoding.
        # NOTE: Current setup for adding position encoding after patch Embedding.
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, input_map, returns_attns=False):
        '''
        The input of the Encoder should be of dim (b, c, h, w).
        :param input_map: The input map for planning.
        :param returns_attns: If True, the model returns slf_attns at each layer
        '''
        enc_output = self.to_patch_embedding(input_map)
        conv_map_shape = enc_output.shape[-2:]
        enc_output = self.reorder_dims(enc_output)

        enc_output = self.position_enc(enc_output, conv_map_shape)

        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)
        return enc_output

# class EnvEncoder(nn.Module):
#     ''' The environment encoder of the planner.
#     '''

#     def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, dropout, n_position):
#         '''
#         Intialize the encoder.
#         :param n_layers: Number of layers of attention and fully connected layer.
#         :param n_heads: Number of self attention modules.
#         :param d_k: Dimension of each Key.
#         :param d_v: Dimension of each Value.
#         :param d_model: Dimension of input/output of encoder layer.
#         :param d_inner: Dimension of the hidden layers of position wise FFN
#         :param dropout: The value to the dropout argument.
#         :param n_position: Total number of patches the model can handle.
#         :param train_shape: The shape of the output of the patch encodings.
#         '''
#         super().__init__()
#         # Convert the image to and input embedding.
#         # NOTE: This is one place where we can add convolution networks.
#         # Convert the image to linear model

#         # NOTE: Padding of 3 is added to the final layer to ensure that
#         # the output of the network has receptive field across the entire map.
#         # NOTE: pytorch doesn't have a good way to ensure automatic padding. This
#         # allows only for a select few map sizes to be solved using this method.
#         self.to_patch_embedding = nn.Sequential(
#             nn.Conv2d(2, 6, kernel_size=5),
#             nn.MaxPool2d(kernel_size=2),
#             nn.ReLU(),
#             nn.Conv2d(6, 16, kernel_size=5),
#             nn.MaxPool2d(kernel_size=2),
#             nn.ReLU(),
#             nn.Conv2d(16, d_model, kernel_size=5, stride=5, padding=3)
#         )

#         self.reorder_dims = Rearrange('b c h w -> b (h w) c')
#         # Position Encoding.
#         # NOTE: Current setup for adding position encoding after patch Embedding.
#         self.position_enc = PositionalEncoding(d_model, n_position=n_position)

#         self.dropout = nn.Dropout(p=dropout)
#         self.layer_stack = nn.ModuleList([
#             EncoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
#             for _ in range(n_layers)
#         ])

#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

#     def forward(self, input_map, returns_attns=False):
#         '''
#         The input of the Encoder should be of dim (b, c, h, w).
#         :param input_map: The input map for planning.
#         :param returns_attns: If True, the model returns slf_attns at each layer
#         '''
#         enc_slf_attn_list = []
#         enc_output = self.to_patch_embedding(input_map)
#         conv_map_shape = enc_output.shape[-2:]
#         enc_output = self.reorder_dims(enc_output)

#         if self.training:
#             enc_output = self.position_enc(enc_output)
#         else:
#             enc_output = self.position_enc(enc_output, conv_map_shape)

#         enc_output = self.dropout(enc_output)
#         enc_output = self.layer_norm(enc_output)

#         for enc_layer in self.layer_stack:
#             enc_output = enc_layer(enc_output, slf_attn_mask=None)

#         if returns_attns:
#             return enc_output, enc_slf_attn_list
#         return enc_output,
