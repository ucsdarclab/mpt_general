# Define the vector quantizer module.
# Taken from - https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
#           and https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py

import torch
import torch.nn as nn

from torch import einsum
from einops import rearrange


class VectorQuantizer(nn.Module):
    ''' A vector quantizer for storing the dictionary of sample points.
    '''

    def __init__(self, n_e, e_dim):
        '''
        :param n_e: Number of elements in the emebedding.
        :param e_dim: Size of the latent embedding vector.
        :param beta: 
        '''
        super().__init__()

        self.n_e = n_e
        self.e_dim = e_dim

        # Initialize the embedding.
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
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
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened,
                         rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        perplexity = None
        min_encodings = None

        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, (perplexity, min_encodings, min_encoding_indices)
