''' VAE style decoder.
'''
import torch.nn as nn

from modules.SubLayers import PositionwiseFeedForward, MultiHeadAttention
from modules.SubLayers import PositionwiseFeedForwardPreNorm, MultiHeadAttentionPreNorm

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        '''
        Initialize the Layer
        :param d_model: Dimension of input/output this layer.
        :param d_inner: Dimension of hidden layer of the position wise FFN
        :param n_head: Number of self-attention modules.
        :param d_k: Dimension of each Key.
        :param d_v: Dimension of each Value.
        :param dropout: Argument to the dropout layer.
        '''
        super(DecoderLayer, self).__init__()
        self.enc_attn = MultiHeadAttentionPreNorm(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardPreNorm(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        '''
        Callback function
        :param dec_input:
        :param enc_output:
        :param slf_attn_mask:
        :param dec_enc_attn_mask:
        '''
        dec_output = self.enc_attn(dec_input, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output


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

class DecoderPreNorm(nn.Module):
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
        self.mu = nn.Sequential(
            nn.Linear(e_dim, c_space_dim),
            nn.Tanh()
        )
        self.sigma = nn.Sequential(
            nn.Linear(e_dim, c_space_dim),
        )

    def forward(self, z_q):
        ''' Returns the decoded mean and variance.
        :param z_q: Latent encoding vectors.
        :returns tuple: mean and diagonal variance vectors.
        '''
        z_q = self.pos_ffn(z_q)
        var = F.softplus(self.sigma(z_q))
        return (self.mu(z_q)+1)/2, var