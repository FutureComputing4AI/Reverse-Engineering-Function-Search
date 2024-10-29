import sys
from typing import Any, Callable

import jax
from jax import numpy as jnp

import flax
from flax import linen as nn
from flax import struct

"""
Default hyperparams taken from flax for now
Except number of layers is changed to 2
"""
@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

  vocab_size: int
  output_vocab_size: int
  dtype: Any = jnp.float32
  num_heads: int = 4 # Original: 8
  qkv_dim: int = 256 # Original: 512
  mlp_dim: int = 512 # Original: 2048
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = True
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """

  config: TransformerConfig
  out_dim: int | None = None

  @nn.compact
  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    config = self.config
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        config.mlp_dim,
        dtype=config.dtype,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
    )(inputs)
    x = nn.relu(x)
    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=config.deterministic
    )
    output = nn.Dense(
        actual_out_dim,
        dtype=config.dtype,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
    )(x)
    output = nn.Dropout(rate=config.dropout_rate)(
        output, deterministic=config.deterministic
    )
    return output


class Encoder1D(nn.Module):

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, encoder_mask=None):
        """
        A single layer 1D Encoder from 
        https://github.com/google/flax/blob/main/examples/wmt/models.py

        Args:
            inputs: input data
            encoder_mask: encoder self-attention mask

        Returns:
            output after transformer encoder block
        """

        config = self.config

        x = nn.LayerNorm(dtype=config.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            qkv_features=config.qkv_dim,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=config.attention_dropout_rate,
            deterministic=config.deterministic,
        )(x, mask=encoder_mask)

        x = nn.Dropout(rate=config.dropout_rate)(
            x, deterministic=config.deterministic
        )
        
        # MLP block.
        y = nn.LayerNorm(dtype=config.dtype)(x)
        y = MlpBlock(config=config)(y)

        return x + y




class BinaryTransformerEncoder(nn.Module):

    channels: int = 128
    embd_size: int = 512
    n_layers = 2
    configs: TransformerConfig = TransformerConfig(
                                    vocab_size=257,
                                    output_vocab_size=None,
                                    emb_dim=512,
                                )

    def _make_embedding_mask(self, x):
        padding_idx = 0
        mask_flat = jnp.expand_dims((jnp.where(x>255, 0, 1)), -1)
        mask = jnp.tile(mask_flat, (1, self.embd_size))
        
        return mask
    
    @nn.compact
    def __call__(self, x):
        mask = self._make_embedding_mask(x)
        x = nn.Embed(257, self.embd_size)(x)
        x = x * mask

        x = Encoder1D(self.configs, name="encoder_1")(x)
        x = Encoder1D(self.configs, name="encoder_2")(x)

        window_shape = (x.shape[1],)
        x = nn.max_pool(x, window_shape, strides=window_shape, padding='VALID')
        reshaped = jnp.reshape(x, (x.shape[0], -1))

        output = nn.Dense(self.channels)(reshaped)

        return output

        

