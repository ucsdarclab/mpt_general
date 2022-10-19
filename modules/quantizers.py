# Define the vector quantizer module.
# Taken from - https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
#           and https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops.layers.torch import Rearrange
from einops import rearrange


class VectorQuantizer(nn.Module):
    ''' A vector quantizer for storing the dictionary of sample points.
    '''

    def __init__(self, n_e, e_dim, latent_dim):
        '''
        :param n_e: Number of elements in the embedding.
        :param e_dim: Size of the latent embedding vector.
        :param latent_dim: Dimension of the encoder vector.
        '''
        super().__init__()

        self.n_e = n_e
        self.e_dim = e_dim

        # Define the linear layer.
        self.input_linear_map = nn.Linear(latent_dim, e_dim)
        self.output_linear_map = nn.Linear(e_dim, latent_dim)

        # Initialize the embedding.
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.batch_norm = nn.BatchNorm1d(self.e_dim, affine=False)
        # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z, mask):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, num_seq, latent_encoding)

        quantization pipeline:
            1. get encoder output (B, S, E)
            2. flatten input to (B*S, E)
        """
        # flatten input vector
        z_flattened = rearrange(z, 'B S E -> (B S) E')
        # pass through the input projection.
        z_flattened = self.input_linear_map(z_flattened)

        # Normalize input vectors.
        z_flattened = F.normalize(z_flattened)
        # Normalize embedding vectors.
        self.embedding.weight.data = F.normalize(self.embedding.weight.data)

        # =========== Apply batch norm =======================

        # # flatten mask
        # mask_flatten = mask.view(-1)

        # z_flattened[mask_flatten == 1, :] = self.batch_norm(
        #     z_flattened[mask_flatten == 1, :])

        # ====================================================

        # # =========== Assuming vectors are NOT normalized ==============
        # # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
        #     torch.sum(self.embedding.weight**2, dim=1) - 2 * \
        #     torch.einsum('bd,dn->bn', z_flattened,
        #                  rearrange(self.embedding.weight, 'n d -> d n'))
        # # ==============================================================

        # =========== Assuming vectors are normalized ==============
        # distances from z to embeddings e_j (z - e)^2 = - e * z
        d = - torch.einsum('bd,dn->bn', z_flattened,
                         rearrange(self.embedding.weight, 'n d -> d n'))
        # ==============================================================

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q_flattened = self.embedding(min_encoding_indices)

        # Translate to output encoder shape
        z_q_flattened = self.output_linear_map(z_q_flattened)
        z_q = z_q_flattened.view(z.shape)

        perplexity = None
        min_encodings = None

        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, (perplexity, min_encodings, min_encoding_indices)


class VQEmbeddingEMA(VectorQuantizer):
    def __init__(self, n_e, e_dim, latent_dim, decay=0.999, epsilon=1e-5):
        ''' Vector quantization for 
        '''
        super(VQEmbeddingEMA, self).__init__(n_e, e_dim, latent_dim)
        self.decay = decay
        self.epsilon = epsilon

        # Don't update weight using gradient, rather use update weight embedding.
        self.embedding.weight.requires_grad = False

        self.register_buffer("ema_count", torch.zeros(n_e)+epsilon)
        self.register_buffer("ema_weight", self.embedding.weight.clone())

    def update_embedding_weights(self, encoding_flatten, one_hot_flatten):
        ''' Update the embedding using EMA.
        :param encoding_flatten: quantized tensor with (BxS)xE shape.
        :param one_hot_flatten: one hot encoding of vector.
        '''
        self.ema_count = self.decay*self.ema_count + \
            (1-self.decay)*one_hot_flatten.sum(axis=0)

        dw = one_hot_flatten.float().T @ encoding_flatten.detach()
        self.ema_weight = self.decay*self.ema_weight + (1-self.decay)*dw

        self.embedding.weight.data = self.ema_weight/self.ema_count[:, None]
