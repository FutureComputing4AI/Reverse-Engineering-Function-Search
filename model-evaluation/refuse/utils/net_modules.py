import sys

import jax
from jax import numpy as jnp

import flax
from flax import linen as nn

class REFUSE(nn.Module):
    """
        Implements MalConv in JAX, following the PyTorch code
        in https://github.com/FutureComputing4AI/MalConv2.

        Removes the final linear layer to produce an embedding,
        rather than a binary prediction.
    """
    
    channels: int = 128
    window_size: int = 8,
    stride: int = 8
    embd_size: int = 8
    log_stride: int = None

    def setup(self):

        self.embd = nn.Embed(257, self.embd_size)
        
        if not self.log_stride is None:
            self.stride = 2**self.log_stride
    
        self.conv_1 = nn.Conv(self.channels, self.window_size,
                                strides=self.stride, use_bias=True)
        self.conv_2 = nn.Conv(self.channels, self.window_size,
                                strides=self.stride, use_bias=True)

        self.fc_1 = nn.Dense(self.channels)

    def _make_embedding_mask(self, x):
        padding_idx = 0
        mask_flat = jnp.expand_dims((jnp.where(x>255, 0, 1)), -1)
        mask = jnp.tile(mask_flat, (1, self.embd_size))
        
        return mask

    @nn.compact
    def __call__(self, x):
        mask = self._make_embedding_mask(x)
        x = self.embd(x)
        x = x * mask
        
        cnn_value = self.conv_1(x)
        gating_weight = nn.sigmoid(self.conv_2(x))
        x = cnn_value * gating_weight
        
        window_shape = (x.shape[1],)
        x = nn.max_pool(x, window_shape, strides=window_shape, padding='VALID')
        
        post_conv = jnp.reshape(x, (x.shape[0], -1))
        ult = self.fc_1(post_conv)
        
        return ult
