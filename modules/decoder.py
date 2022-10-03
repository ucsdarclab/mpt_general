''' VAE style decoder.
'''
import torch.nn as nn

from modules.SubLayers import PositionwiseFeedForward


class Decoder(nn.Module):
    ''' Decoder that takes the latent encoding and generates joint samples.
    '''

    def __init__(self, e_dim, h_dim, c_space_dim, dropout=0.5):
        '''
        :param e_dim: Dimension of the dictionary vectors.
        :param h_dim: Dimension of the feedforward networks hidden vector.
        :param c_space_dim: Dimension of the c-space.
        :param dropout: Dropout value for the fullyconnected layer.
        '''
        super().__init__()
        self.pos_ffn = PositionwiseFeedForward(e_dim, h_dim, dropout)

        # Layers for returning mean and variance
        self.mu = nn.Linear(e_dim, c_space_dim)
        self.sigma = nn.Sequential(
            nn.Linear(e_dim, c_space_dim),
            nn.ReLU()
        )

    def forward(self, z_q):
        ''' Returns the decoded mean and variance.
        :param z_q: Latent encoding vectors.
        :returns tuple: mean and diagonal variance vectors.
        '''
        z_q = self.pos_ffn(z_q)
        return self.mu(z_q), self.sigma(z_q)
