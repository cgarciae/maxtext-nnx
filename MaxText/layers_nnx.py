"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Transformer model definition.
"""
# pylint: disable=arguments-differ

from MaxText.aqt.jax.v2 import aqt_dot_general as aqt
from MaxText.aqt.jax.v2 import config as aqt_config

import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union


from flax.linen import partitioning as nn_partitioning
from flax.experimental import nnx

import numpy as np

import jax
from jax import lax
from jax import random
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp


withLP = nnx.with_logical_partitioning
ScanIn = nn_partitioning.ScanIn

Config = Any

# Type annotations
Array = jax.Array
DType = jnp.dtype
PRNGKey = jax.Array
Shape = Iterable[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = nnx.initializers.Initializer
InitializerAxis = Union[int, Tuple[int, ...]]
NdInitializer = Callable[
    [PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]

default_embed_init = nnx.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0)


#------------------------------------------------------------------------------
# Dot product attention layer.
#------------------------------------------------------------------------------


def dot_product_attention(
  query: Array,
  key: Array,
  value: Array,
  bias: Optional[Array] = None,
  dropout_rng: Optional[PRNGKey] = None,
  dropout_rate: float = 0.,
  deterministic: bool = False,
  dtype: DType = jnp.float32,
  float32_logits: bool = False,
  *,
  query_layer_norm: "LayerNorm",
  key_layer_norm: "LayerNorm",
):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Args:
    query: queries for calculating attention with shape of `[batch, q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch, kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch, kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch, num_heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.

  Returns:
    Output of shape `[batch, length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      'q, k, v num_heads must match.')
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # Casting logits and softmax computation for float32 for model stability.
  if float32_logits:
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
  # Layer norms here prevent (near) one-hot softmaxes, which can lead to
  # unstable training loss and nans, see the "QK Normalization" subsection in
  # https://arxiv.org/pdf/2302.05442.pdf.
  query = query_layer_norm(query)
  key = key_layer_norm(key)

  # `attn_weights`: [batch, num_heads, q_length, kv_length]
  attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)

  # Apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias.astype(attn_weights.dtype)

  # Normalize the attention weights across `kv_length` dimension.
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # Apply attention dropout.
  if not deterministic and dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    # Broadcast dropout mask along the query dim.
    dropout_shape = list(attn_weights.shape)
    dropout_shape[-2] = 1
    keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    keep = jnp.broadcast_to(keep, attn_weights.shape)
    multiplier = (
        keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  # Take the linear combination of `value`.
  return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)


dynamic_vector_slice_in_dim = jax.vmap(
    lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))


#------------------------------------------------------------------------------
# DenseGeneral for attention layers.
#------------------------------------------------------------------------------


def nd_dense_init(scale, mode, distribution):
  """Initializer with in_axis, out_axis set at call time."""
  def init_fn(key, shape, dtype, in_axis, out_axis):
    fn = jax.nn.initializers.variance_scaling(
        scale, mode, distribution, in_axis, out_axis)
    return fn(key, shape, dtype)
  return init_fn


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)

class DenseGeneral(nnx.Module):
  """A linear transformation (without bias) with flexible axes.

    Attributes:
      features: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
  """

  def __init__(
    self,
    features_in: Union[Iterable[int], int],
    features_out: Union[Iterable[int], int],
    *,
    config: Config,
    axis: Union[Iterable[int], int] = -1,
    dtype: DType = jnp.float32,
    kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'truncated_normal'),
    kernel_axes: Tuple[str, ...] = (),
    ctx: nnx.Context,
  ):
    self.features_in = features_in
    self.features_out = features_out
    self.axis = axis
    self.config = config
    self.dtype = dtype
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes

    features_in = _canonicalize_tuple(self.features_in)
    features_out = _canonicalize_tuple(self.features_out)
    axis = _canonicalize_tuple(self.axis)

    kernel_shape = features_in + features_out
    kernel_in_axis = np.arange(len(axis))
    kernel_out_axis = np.arange(len(axis), len(axis) + len(features_out))
    self.kernel = nnx.Param(
      withLP(
        self.kernel_init,
        sharding=self.kernel_axes,
      )(
        ctx.make_rng('params'),
        kernel_shape,
        jnp.float32,
        kernel_in_axis,
        kernel_out_axis,
      )
    )

  def __call__(self, inputs: Array, *, ctx: nnx.Context) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    config = self.config
    axis = _canonicalize_tuple(self.axis)
    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)
    kernel = jnp.asarray(self.kernel, self.dtype)
    contract_ind = tuple(range(0, len(axis)))

    if not config.use_int8_training:
      return lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))
    else:
      aqt_cfg = aqt_config.fully_quantized(bits=8, use_fwd_quant=True)

      def noise_fn(shape, key):
        return jax.random.uniform(key, shape) - 0.5

      aqt_cfg.dlhs.lhs.noise_fn = noise_fn
      aqt_cfg.dlhs.rhs.noise_fn = noise_fn
      aqt_cfg.drhs.lhs.noise_fn = noise_fn
      aqt_cfg.drhs.rhs.noise_fn = noise_fn

      aqt_dot_general = aqt.make_dot_general(aqt_cfg)
      aqt_key = ctx.make_rng('aqt')
      context = aqt.Context(key=aqt_key, train_step=None)

      return aqt_dot_general(
        inputs, kernel, ((axis, contract_ind), ((), ())), context=context)

def _convert_to_activation_function(
    fn_or_string: Union[str, Callable]) -> Callable:
  """Convert a string to an activation function."""
  if fn_or_string == 'linear':
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nnx, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError(f"""Don't know how to convert {fn_or_string}
                         to an activation function""")


class MultiHeadDotProductAttention(nnx.Module):
  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      head_dim: dimension of each head.
      dtype: the dtype of the computation.
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
  """

  def __init__(
    self,
    emb_dim: int,
    num_heads: int,
    head_dim: int,
    *,
    config: Config,
    dtype: DType = jnp.float32,
    dropout_rate: float = 0.0,
    kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'normal'),
    float32_logits: bool = False,  # computes logits in float32 for stability.
    ctx: nnx.Context,
  ):
    self.emb_dim = emb_dim
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.config = config
    self.dtype = dtype
    self.dropout_rate = dropout_rate
    self.kernel_init = kernel_init
    self.float32_logits = float32_logits

    # NOTE: T5 does not explicitly rescale the attention logits by
    # 1/sqrt(depth_kq)!  This is folded into the initializers of the
    # linear transformations, which is equivalent under Adafactor.
    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    
    def query_init(*args):
      #pylint: disable=no-value-for-parameter
      return self.kernel_init(*args) / depth_scaling

    projection = functools.partial(
      DenseGeneral,
      features_in=self.emb_dim,
      features_out=(self.num_heads, self.head_dim),
      axis=-1,
      kernel_axes=('embed', 'heads', 'kv'),
      dtype=self.dtype,
      config=config,
      ctx=ctx,
    )
    
    self.query = projection(kernel_init=query_init)
    self.key = projection(kernel_init=self.kernel_init)
    self.value = projection(kernel_init=self.kernel_init)
    self.out = DenseGeneral(
      features_in=(self.num_heads, self.head_dim),
      features_out=self.emb_dim,  # output dim is set to the input dim.
      axis=(-2, -1),
      kernel_init=self.kernel_init,
      kernel_axes=('heads', 'kv', 'embed'),
      dtype=self.dtype,
      config=config,
      ctx=ctx,
    )
    self.query_layer_norm = LayerNorm(
      self.head_dim, dtype=dtype, kernel_axes = ('heads',), ctx=ctx
    )
    self.key_layer_norm = LayerNorm(
      self.head_dim, dtype=dtype, kernel_axes = ('heads',), ctx=ctx
    )

  def init_cache(self, batch_size: int, seq_len: int):
    cache_shape = (batch_size, self.num_heads, self.head_dim, seq_len)
    index_shape = ()
    
    if self.config.scan_layers:
      cache_shape = (self.config.num_decoder_layers, *cache_shape)
      index_shape = (self.config.num_decoder_layers,)

    self.cached_key = nnx.Cache(jnp.zeros(cache_shape, self.dtype))
    self.cached_value = nnx.Cache(jnp.zeros(cache_shape, self.dtype))
    self.cache_index = nnx.Cache(jnp.zeros(index_shape, dtype=jnp.int32))

  def __call__(
    self,
    inputs_q: Array,
    inputs_kv: Array,
    mask: Optional[Array] = None,
    bias: Optional[Array] = None,
    *,
    decode: bool = False,
    deterministic: bool = False,
    ctx: nnx.Context
  ) -> Array:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    There are two modes: decoding and non-decoding (e.g., training). The mode is
    determined by `decode` argument. For decoding, this method is called twice,
    first to initialize the cache and then for an actual decoding process. The
    two calls are differentiated by the presence of 'cached_key' in the variable
    dict. In the cache initialization stage, the cache variables are initialized
    as zeros and will be filled in the subsequent decoding process.

    In the cache initialization call, `inputs_q` has a shape [batch, length,
    q_features] and `inputs_kv`: [batch, length, kv_features]. During the
    incremental decoding stage, query, key and value all have the shape [batch,
    1, qkv_features] corresponding to a single step.

    Args:
      inputs_q: input queries of shape `[batch, q_length, q_features]`.
      inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
      mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
      bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
      decode: Whether to prepare and use an autoregressive cache.
      deterministic: Disables dropout if set to True.

    Returns:
      output of shape `[batch, length, q_features]`.
    """
    # Project inputs_q to multi-headed q/k/v
    # dimensions are then [batch, length, num_heads, head_dim]
    query = self.query(inputs_q, ctx=ctx)
    key = self.key(inputs_kv, ctx=ctx)
    value = self.value(inputs_kv, ctx=ctx)

    query = nnx.with_logical_constraint(
        query, ('activation_batch', 'activation_length', 'activation_heads', 'activation_kv')
    )
    query = checkpoint_name(query, 'query_proj')
    key = nnx.with_logical_constraint(key, ('activation_batch', 'activation_length', 'activation_heads', 'activation_kv'))
    key = checkpoint_name(key, 'key_proj')
    value = nnx.with_logical_constraint(
        value, ('activation_batch', 'activation_length', 'activation_heads', 'activation_kv')
    )
    value = checkpoint_name(value, 'value_proj')

    if decode:
      # The key and value have dimension [batch, length, num_heads, head_dim],
      # but we cache them as [batch, num_heads, head_dim, length] as a TPU
      # fusion optimization. This also enables the "scatter via one-hot
      # broadcast" trick, which means we do a one-hot broadcast instead of a
      # scatter/gather operations, resulting in a 3-4x speedup in practice.

      batch, num_heads, head_dim, length = self.cached_key.shape
      # During fast autoregressive decoding, we feed one position at a time,
      # and cache the keys and values step by step.
      # Sanity shape check of cached key against input query.
      expected_shape = (batch, 1, num_heads, head_dim)
      if expected_shape != query.shape:
        raise ValueError(f"""Autoregressive cache shape error,
                          expected query shape %s instead got 
                          {(expected_shape, query.shape)}""")
      # Create a OHE of the current index. NOTE: the index is increased below.
      cur_index = self.cache_index
      one_hot_indices = jax.nn.one_hot(cur_index, length, dtype=key.dtype)
      # In order to update the key, value caches with the current key and
      # value, we move the length axis to the back, similar to what we did for
      # the cached ones above.
      # Note these are currently the key and value of a single position, since
      # we feed one position at a time.
      one_token_key = jnp.moveaxis(key, -3, -1)
      one_token_value = jnp.moveaxis(value, -3, -1)
      # Update key, value caches with our new 1d spatial slices.
      # We implement an efficient scatter into the cache via one-hot
      # broadcast and addition.
      key = self.cached_key + one_token_key * one_hot_indices
      value = self.cached_value + one_token_value * one_hot_indices
      self.cached_key = key
      self.cached_value = value
      self.cache_index += 1
      # Move the keys and values back to their original shapes.
      key = jnp.moveaxis(key, -1, -3)
      value = jnp.moveaxis(value, -1, -3)

      # Causal mask for cached decoder self-attention: our single query
      # position should only attend to those key positions that have already
      # been generated and cached, not the remaining zero elements.
      mask = combine_masks(
          mask,
          jnp.broadcast_to(
              jnp.arange(length) <= cur_index,
              # (1, 1, length) represent (head dim, query length, key length)
              # query length is 1 because during decoding we deal with one
              # index.
              # The same mask is applied to all batch elements and heads.
              (batch, 1, 1, length)))

      # Grab the correct relative attention bias during decoding. This is
      # only required during single step decoding.
      if bias is not None:
        # The bias is a full attention matrix, but during decoding we only
        # have to take a slice of it.
        # This is equivalent to bias[..., cur_index:cur_index+1, :].
        bias = dynamic_vector_slice_in_dim(
            jnp.squeeze(bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2)

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype))
    else:
      attention_bias = None

    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = combine_biases(attention_bias, bias)

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = ctx.make_rng('dropout')

    # Apply attention.
    x = dot_product_attention(
      query,
      key,
      value,
      bias=attention_bias,
      dropout_rng=dropout_rng,
      dropout_rate=self.dropout_rate,
      deterministic=deterministic,
      dtype=self.dtype,
      float32_logits=self.float32_logits,
      query_layer_norm=self.query_layer_norm,
      key_layer_norm=self.key_layer_norm,
    )

    # Back to the original inputs dimensions.
    out = self.out(x, ctx=ctx)
    return out


class MlpBlock(nnx.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: Type for the dense layer.
  """
  def __init__(
    self,
    emb_dim: int,
    *,
    config: Config,
    intermediate_dim: int = 2048,
    activations: Sequence[Union[str, Callable]] = ('relu',),
    kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'truncated_normal'),
    intermediate_dropout_rate: float = 0.1,
    dtype: Any = jnp.float32,
    ctx: nnx.Context,
  ):
    config = config
    self.emb_dim = emb_dim
    self.intermediate_dim = intermediate_dim
    self.activations = tuple(
      _convert_to_activation_function(a) for a in activations
    )
    self.kernel_init = kernel_init
    self.intermediate_dropout_rate = intermediate_dropout_rate
    self.dtype = dtype

    dense_names = []

    for idx in range(len(self.activations)):
      name = 'wi' if len(self.activations) == 1 else f'wi_{idx}'
      dense = DenseGeneral(
        self.emb_dim if idx == 0 else self.intermediate_dim,
        self.intermediate_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        kernel_axes=('embed', 'mlp'),
        config=config,
        ctx=ctx,
      )
      setattr(self, name, dense)
      dense_names.append(name)

    self.dense_names = tuple(dense_names)
    self.dropout = nnx.Dropout(
      rate=self.intermediate_dropout_rate,
      broadcast_dims=(-2,),
    )
    self.wo = DenseGeneral(
      self.intermediate_dim,
      self.emb_dim,
      dtype=self.dtype,
      kernel_init=self.kernel_init,
      kernel_axes=('mlp', 'embed'),
      config=config,
      ctx=ctx,
    )



  def __call__(
    self, inputs, *, decode: bool = False, deterministic: bool = False, ctx: nnx.Context):
    """Applies Transformer MlpBlock module."""
    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    for name, act_fn in zip(self.dense_names, self.activations):
      dense: DenseGeneral = getattr(self, name)
      x = dense(inputs, ctx=ctx)
      x = act_fn(x)
      activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations)
    # Apply dropout and final dense output projection.
    x = self.dropout(x, deterministic=deterministic)  # Broadcast along length.
    x = nnx.with_logical_constraint(x, ('activation_batch', 'activation_length', 'activation_mlp'))
    output = self.wo(x, ctx=ctx)
    return output



#------------------------------------------------------------------------------
# T5 Layernorm - no subtraction of mean or bias.
#------------------------------------------------------------------------------

class LayerNorm(nnx.Module):
  """T5 Layer normalization operating on the last axis of the input data."""

  def __init__(
    self,
    features: int,
    *,
    epsilon: float = 1e-6,
    dtype: Any = jnp.float32,
    kernel_axes: Tuple[str, ...] = (),
    scale_init: Initializer = nnx.initializers.ones(),
    ctx: nnx.Context,
  ):
    self.features = features
    self.epsilon = epsilon
    self.dtype = dtype
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init
    
    self.scale = nnx.Param(
      withLP(
        self.scale_init,
        sharding=self.kernel_axes,
      )(
        ctx.make_rng('params'),
        (features,), 
        jnp.float32
      )
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = jnp.asarray(self.scale, self.dtype)
    return y * scale


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

class Embed(nnx.Module):
  """A parameterized function from integers [0, n) to d-dimensional vectors.

  Attributes:
    num_embeddings: number of embeddings.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: float32).
    embedding_init: embedding initializer.
  """
  def __init__(
    self,
    num_embeddings: int,
    features: int,
    *,
    cast_input_dtype: Optional[DType] = None,
    dtype: DType = jnp.float32,
    attend_dtype: Optional[DType] = None,
    embedding_init: Initializer = default_embed_init,
    ctx: nnx.Context,
  ):
    self.num_embeddings = num_embeddings
    self.features = features
    self.cast_input_dtype = cast_input_dtype
    self.dtype = dtype
    self.attend_dtype = attend_dtype
    self.embedding_init = embedding_init

    self.embedding = nnx.Param(
      withLP(
        self.embedding_init,
        sharding=('vocab', 'embed'),
      )(
        ctx.make_rng('params'),
        (self.num_embeddings, self.features),
        jnp.float32
      ),
    )

  def __call__(self, inputs: Array) -> Array:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `features` dimension appended.
    """
    if self.cast_input_dtype:
      inputs = inputs.astype(self.cast_input_dtype)
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError('Input type must be an integer or unsigned integer.')
    output = jnp.asarray(self.embedding, self.dtype)[inputs]
    output = nnx.with_logical_constraint(output, ('activation_batch', 'activation_length', 'activation_embed'))
    return output

  def attend(self, query: Array) -> Array:
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `features` of the
        embedding.

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
    return jnp.dot(query, jnp.asarray(self.embedding, dtype).T)

class RelativePositionBiases(nnx.Module):
  """Adds T5-style relative positional embeddings to the attention logits.

  Attributes:
    num_buckets: Number of buckets to bucket distances between key and query
      positions into.
    max_distance: Maximum distance before everything is lumped into the last
      distance bucket.
    num_heads: Number of heads in the attention layer. Each head will get a
      different relative position weighting.
    dtype: Type of arrays through this module.
    embedding_init: initializer for relative embedding table.
  """
  def __init__(
    self,
    num_buckets: int,
    max_distance: int,
    num_heads: int,
    dtype: Any,
    embedding_init: Callable[..., Array] = default_embed_init,
    *,
    ctx: nnx.Context,
  ):
    self.num_buckets = num_buckets
    self.max_distance = max_distance
    self.num_heads = num_heads
    self.dtype = dtype
    self.embedding_init = embedding_init
    self.rel_embedding = nnx.Param(
      withLP(
        self.embedding_init,
        sharding=('heads', 'relpos_buckets'),
      )(
        ctx.make_rng('params'),
        (self.num_heads, self.num_buckets),
        jnp.float32,
      ),
    )

  @staticmethod
  def _relative_position_bucket(relative_position,
                                bidirectional=True,
                                num_buckets=32,
                                max_distance=128):
    """Translate relative position to a bucket number for relative attention.

    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position.  If bidirectional=False, then positive relative positions are
    invalid.
    We use smaller buckets for small absolute relative_position and larger
    buckets for larger absolute relative_positions.  All relative
    positions >=max_distance  map to the same bucket.  All relative
    positions <=-max_distance map to the same bucket.  This should allow for
    more graceful generalization to longer sequences than the model has been
    trained on.

    Args:
      relative_position: an int32 array
      bidirectional: a boolean - whether the attention is bidirectional
      num_buckets: an integer
      max_distance: an integer

    Returns:
      a Tensor with the same shape as relative_position, containing int32
        values in the range [0, num_buckets)
    """
    ret = 0
    n = -relative_position
    if bidirectional:
      num_buckets //= 2
      ret += (n < 0).astype(np.int32) * num_buckets
      n = np.abs(n)
    else:
      n = np.maximum(n, 0)
    # now n is in the range [0, inf)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_if_large = max_exact + (
        np.log(n.astype(np.float32) / max_exact + np.finfo(np.float32).eps) /
        np.log(max_distance / max_exact) *
        (num_buckets - max_exact)).astype(np.int32)
    val_if_large = np.minimum(val_if_large, num_buckets - 1)
    ret += np.where(is_small, n, val_if_large)
    return ret

  def __call__(self, qlen, klen, bidirectional=True):
    """Produce relative position embedding attention biases.

    Args:
      qlen: attention query length.
      klen: attention key length.
      bidirectional: whether to allow positive memory-query relative position
        embeddings.

    Returns:
      output: `(1, len, q_len, k_len)` attention bias
    """
    # TODO: should we be computing this w. numpy as a program
    # constant?
    context_position = np.arange(qlen, dtype=jnp.int32)[:, None]
    memory_position = np.arange(klen, dtype=jnp.int32)[None, :]
    relative_position = memory_position - context_position  # shape (qlen, klen)
    rp_bucket = self._relative_position_bucket(
        relative_position,
        bidirectional=bidirectional,
        num_buckets=self.num_buckets,
        max_distance=self.max_distance)
    relative_attention_bias = self.rel_embedding

    relative_attention_bias = jnp.asarray(relative_attention_bias, self.dtype)
    # Instead of using a slow gather, we create a leading-dimension one-hot
    # array from rp_bucket and use it to perform the gather-equivalent via a
    # contraction, i.e.:
    # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
    # This is equivalent to relative_attention_bias[:, rp_bucket]
    bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1), 0)
    rp_bucket_one_hot = jnp.array(
        rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype)
    # --> shape (qlen, klen, num_heads)
    values = lax.dot_general(
        relative_attention_bias,
        rp_bucket_one_hot,
        (
            ((1,), (0,)),  # rhs, lhs contracting dims
            ((), ())))  # no batched dims
    # Add a singleton batch dimension.
    # --> shape (1, num_heads, qlen, klen)
    return values[jnp.newaxis, ...]


#------------------------------------------------------------------------------
# Mask-making utility functions.
#------------------------------------------------------------------------------
def make_attention_mask(
  query_input: Array,
  key_input: Array,
  pairwise_fn: Callable = jnp.multiply,
  extra_batch_dims: int = 0,
  dtype: DType = jnp.float32,
) -> Array:
  """Mask-making helper for attention weights.

  In case of 1d inputs (i.e., `[batch, len_q]`, `[batch, len_kv]`, the
  attention weights will be `[batch, heads, len_q, len_kv]` and this
  function will produce `[batch, 1, len_q, len_kv]`.

  Args:
    query_input: a batched, flat input of query_length size
    key_input: a batched, flat input of key_length size
    pairwise_fn: broadcasting elementwise comparison function
    extra_batch_dims: number of extra batch dims to add singleton axes for, none
      by default
    dtype: mask return dtype

  Returns:
    A `[batch, 1, len_q, len_kv]` shaped mask for 1d attention.
  """
  # [batch, len_q, len_kv]
  mask = pairwise_fn(
    # [batch, len_q] -> [batch, len_q, 1]
    jnp.expand_dims(query_input, axis=-1),
    # [batch, len_q] -> [batch, 1, len_kv]
    jnp.expand_dims(key_input, axis=-2)
  )

  # [batch, 1, len_q, len_kv]. This creates the head dim.
  mask = jnp.expand_dims(mask, axis=-3)
  mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
  return mask.astype(dtype)


def make_causal_mask(x: Array,
                     extra_batch_dims: int = 0,
                     dtype: DType = jnp.float32) -> Array:
  """Make a causal mask for self-attention.

  In case of 1d inputs (i.e., `[batch, len]`, the self-attention weights
  will be `[batch, heads, len, len]` and this function will produce a
  causal mask of shape `[batch, 1, len, len]`.

  Note that a causal mask does not depend on the values of x; it only depends on
  the shape. If x has padding elements, they will not be treated in a special
  manner.

  Args:
    x: input array of shape `[batch, len]`
    extra_batch_dims: number of batch dims to add singleton axes for, none by
      default
    dtype: mask return dtype

  Returns:
    A `[batch, 1, len, len]` shaped causal mask for 1d attention.
  """
  idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
  return make_attention_mask(
      idxs,
      idxs,
      jnp.greater_equal,
      extra_batch_dims=extra_batch_dims,
      dtype=dtype)


def combine_masks(*masks: Optional[Array], dtype: DType = jnp.float32):
  """Combine attention masks.

  Args:
    *masks: set of attention mask arguments to combine, some can be None.
    dtype: final mask dtype

  Returns:
    Combined mask, reduced by logical and, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
      f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = jnp.logical_and(mask, other_mask)
  return mask.astype(dtype)


def combine_biases(*masks: Optional[Array]):
  """Combine attention biases.

  Args:
    *masks: set of attention bias arguments to combine, some can be None.

  Returns:
    Combined mask, reduced by summation, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
      f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = mask + other_mask
  return mask


def make_decoder_mask(decoder_target_tokens: Array,
                      dtype: DType,
                      decoder_causal_attention: Optional[Array] = None,
                      decoder_segment_ids: Optional[Array] = None) -> Array:
  """Compute the self-attention mask for a decoder.

  Decoder mask is formed by combining a causal mask, a padding mask and an
  optional packing mask. If decoder_causal_attention is passed, it makes the
  masking non-causal for positions that have value of 1.

  A prefix LM is applied to a dataset which has a notion of "inputs" and
  "targets", e.g., a machine translation task. The inputs and targets are
  concatenated to form a new target. `decoder_target_tokens` is the concatenated
  decoder output tokens.

  The "inputs" portion of the concatenated sequence can attend to other "inputs"
  tokens even for those at a later time steps. In order to control this
  behavior, `decoder_causal_attention` is necessary. This is a binary mask with
  a value of 1 indicating that the position belonged to "inputs" portion of the
  original dataset.

  Example:

    Suppose we have a dataset with two examples.

    ds = [{"inputs": [6, 7], "targets": [8]},
          {"inputs": [3, 4], "targets": [5]}]

    After the data preprocessing with packing, the two examples are packed into
    one example with the following three fields (some fields are skipped for
    simplicity).

       decoder_target_tokens = [[6, 7, 8, 3, 4, 5, 0]]
         decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]
    decoder_causal_attention = [[1, 1, 0, 1, 1, 0, 0]]

    where each array has [batch, length] shape with batch size being 1. Then,
    this function computes the following mask.

                      mask = [[[[1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0]]]]

    mask[b, 1, :, :] represents the mask for the example `b` in the batch.
    Because mask is for a self-attention layer, the mask's shape is a square of
    shape [query length, key length].

    mask[b, 1, i, j] = 1 means that the query token at position i can attend to
    the key token at position j.

  Args:
    decoder_target_tokens: decoder output tokens. [batch, length]
    dtype: dtype of the output mask.
    decoder_causal_attention: a binary mask indicating which position should
      only attend to earlier positions in the sequence. Others will attend
      bidirectionally. [batch, length]
    decoder_segment_ids: decoder segmentation info for packed examples. [batch,
      length]

  Returns:
    the combined decoder mask.
  """
  masks = []
  # The same mask is applied to all attention heads. So the head dimension is 1,
  # i.e., the mask will be broadcast along the heads dim.
  # [batch, 1, length, length]
  causal_mask = make_causal_mask(decoder_target_tokens, dtype=dtype)

  # Positions with value 1 in `decoder_causal_attneition` can attend
  # bidirectionally.
  if decoder_causal_attention is not None:
    # [batch, 1, length, length]
    inputs_mask = make_attention_mask(
        decoder_causal_attention,
        decoder_causal_attention,
        jnp.logical_and,
        dtype=dtype)
    masks.append(jnp.logical_or(causal_mask, inputs_mask).astype(dtype))
  else:
    masks.append(causal_mask)

  # Padding mask.
  masks.append(
      make_attention_mask(
          decoder_target_tokens > 0, decoder_target_tokens > 0, dtype=dtype))

  # Packing mask
  if decoder_segment_ids is not None:
    masks.append(
        make_attention_mask(
            decoder_segment_ids, decoder_segment_ids, jnp.equal, dtype=dtype))

  return combine_masks(*masks, dtype=dtype)



#------------------------------------------------------------------------------
# The network: Decoder & Transformer Definitions
#------------------------------------------------------------------------------


class DecoderLayer(nnx.Module):
  """Transformer decoder layer that attends to the encoder."""

  def __init__(self, config: Config, *, ctx: nnx.Context):
    self.config = config

    self.self_attention = MultiHeadDotProductAttention(
      emb_dim=config.emb_dim,
      num_heads=config.num_heads,
      head_dim=config.head_dim,
      dtype=config.dtype,
      dropout_rate=config.dropout_rate,
      config=config,
      ctx=ctx,
    )
    self.pre_self_attention_layer_norm = LayerNorm(
      config.emb_dim,
      dtype=config.dtype, 
      kernel_axes=('embed',),
      ctx=ctx,
    )
    self.relpos_bias = RelativePositionBiases(
      num_buckets=32,
      max_distance=128,
      num_heads=config.num_heads,
      dtype=config.dtype,
      embedding_init=nnx.initializers.variance_scaling(
        1.0, 'fan_avg', 'uniform'
      ),
      ctx=ctx,
    )
    self.mlp = MlpBlock(
      config.emb_dim,
      intermediate_dim=config.mlp_dim,
      activations=config.mlp_activations,
      intermediate_dropout_rate=config.dropout_rate,
      dtype=config.dtype,
      config=config,
      ctx=ctx,
    )
    self.dropout = nnx.Dropout(
      rate=config.dropout_rate, 
      broadcast_dims=(-2,),
    )

  def __call__(
    self,
    inputs: Array,
    _scan_axes,
    decoder_mask: Optional[Array],
    deterministic: bool,
    decode: bool,
    max_decode_length,
    *,
    ctx: nnx.Context,
  ):
    config = self.config

    # Relative position embedding as attention biases.
    l = max_decode_length if decode and max_decode_length else inputs.shape[-2]
    decoder_bias = self.relpos_bias(l, l, False)

    inputs = nnx.with_logical_constraint(inputs, ('activation_batch', 'activation_length', 'activation_embed'))

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    lnx = self.pre_self_attention_layer_norm(inputs)
    lnx = nnx.with_logical_constraint(lnx, ('activation_batch', 'activation_length', 'activation_embed'))

    # Self-attention block
    attention_lnx = self.self_attention(
      lnx,
      lnx,
      decoder_mask,
      decoder_bias,
      deterministic=deterministic,
      decode=decode,
      ctx=ctx,
    )
    attention_lnx = nnx.with_logical_constraint(attention_lnx, ('activation_batch', 'activation_length', 'activation_embed'))

    # MLP block.
    mlp_lnx = self.mlp(lnx, deterministic=deterministic, ctx=ctx)
    mlp_lnx = nnx.with_logical_constraint(mlp_lnx, ('activation_batch', 'activation_length', 'activation_embed'))

    next_layer_addition = mlp_lnx + attention_lnx

    next_layer_addition_dropped_out = self.dropout(
      next_layer_addition, deterministic=deterministic
    )

    layer_output = next_layer_addition_dropped_out + inputs
    layer_output = nnx.with_logical_constraint(layer_output, ('activation_batch', 'activation_length', 'activation_embed'))

    if config.record_internal_nn_metrics:
      self.sow(nnx.Intermediate, 'activation_mean', jnp.mean(layer_output))
      self.sow(nnx.Intermediate, 'activation_stdev', jnp.std(layer_output))
      self.sow(nnx.Intermediate, 'activation_fraction_zero', jnp.sum(layer_output==0) / jnp.size(layer_output))
    
    return layer_output, None



class Decoder(nnx.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""
  def __init__(
    self, 
    config: Config, 
    *, 
    shared_embedding: Embed, 
    ctx: nnx.Context
  ):
    self.config = config
    self.shared_embedding = shared_embedding

    params_spec = config.param_scan_axis
    cache_spec = 0
    
    self.dropout = nnx.Dropout(
      rate=config.dropout_rate, broadcast_dims=(-2,)
    )

    BlockLayer = DecoderLayer

    if config.remat_policy != 'none':
      if config.remat_policy == 'minimal':
        policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
      elif config.remat_policy == 'proj':
        policy = jax.checkpoint_policies.save_only_these_names(
            'query_proj', 'value_proj', 'key_proj'
        )
      else:
        assert config.remat_policy == 'full', "Remat policy needs to be on list of remat policies"
        policy = None

      BlockLayer = nnx.Remat(
        BlockLayer, 
        policy=policy,
        prevent_cse=not config.scan_layers,
        static_argnums=(-1, -2, -3, -4),
      )

    if config.scan_layers:
      self.layers = nnx.Scan(
        BlockLayer,
        variable_axes={
          nnx.Param: params_spec,
          nnx.Cache: cache_spec,
          nnx.Intermediate: 0
        },
        split_rngs={
          'params': True,
          'dropout': config.enable_dropout,
          'aqt': config.use_int8_training,
        },
        # in_axes=(nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
        length=config.num_decoder_layers,
        metadata_params={nnx.PARTITION_NAME: 'layers'}
      )(config, ctx=ctx)

    else:
      self.layers = nnx.Sequence(
        BlockLayer(config, ctx=ctx)
        for _ in range(config.num_decoder_layers)
      )
    self.decoder_norm = LayerNorm(
      config.emb_dim, dtype=config.dtype, kernel_axes=('embed',), ctx=ctx
    )
    if config.logits_via_embedding:
      self.logits_dense = None
    else:
      self.logits_dense = DenseGeneral(
        config.emb_dim,
        config.vocab_size,
        dtype=jnp.float32,  # Use float32 for stabiliity.
        kernel_axes=('embed', 'vocab'),
        config=config,
        ctx=ctx,
      )
  

  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions=None,
      decoder_mask=None,
      deterministic=False,
      decode=False,
      max_decode_length=None,
      *,
      ctx: nnx.Context,
    ):
    assert decoder_input_tokens.ndim == 2  # [batch, len]
    config = self.config 

    # [batch, length] -> [batch, length, emb_dim]
    y = self.shared_embedding(decoder_input_tokens.astype('int32'))
    y = self.dropout(y, deterministic=deterministic).astype(config.dtype)
    
    y, _ = self.layers(
      y, None, decoder_mask, deterministic, decode, max_decode_length, ctx=ctx
    )

    y = self.decoder_norm(y)
    y = self.dropout(y, deterministic=deterministic, ctx=ctx)

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if self.config.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.shared_embedding.attend(y)
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      assert self.logits_dense is not None
      logits = self.logits_dense(y, ctx=ctx)
    logits = nnx.with_logical_constraint(logits, ('activation_batch', 'activation_length', 'activation_vocab'))
    return logits


class Transformer(nnx.Module):
  """An decoder-only Transformer model."""
  # pylint: disable=attribute-defined-outside-init

  def __init__(
    self,
    config: Config,
    *,
    ctx: nnx.Context
  ):
    self.config = config
    self.token_embedder = Embed(
      num_embeddings=config.vocab_size,
      features=config.emb_dim,
      dtype=config.dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=nnx.initializers.normal(stddev=1.0),
      ctx=ctx,
    )
    self.decoder = Decoder(
      config=config, shared_embedding=self.token_embedder, ctx=ctx
    )

  def __call__(
      self,
      decoder_input_tokens,
      decoder_target_tokens,
      decoder_segment_ids=None,
      decoder_positions=None,
      enable_dropout=True,
      decode=False,
      max_decode_length=None,
      *,
      ctx: nnx.Context,
    ):
    """Applies Transformer decoder-branch on encoded-input and target."""
    config = self.config

    # Make padding attention masks.
    if decode:
      # Do not mask decoder attention based on targets padding at
      # decoding/inference time.
      decoder_mask = None
    else:
      decoder_mask = make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        dtype=config.dtype,
        decoder_segment_ids=decoder_segment_ids
      )

    # Add segmentation block-diagonal attention masks if using segmented data.
    if decoder_segment_ids is not None:
      if decode:
        raise ValueError(
            'During decoding, packing should not be used but '
            '`encoder_segment_ids` was passed to `Transformer.decode`.')

    logits = self.decoder(
      decoder_input_tokens=decoder_input_tokens,
      decoder_positions=decoder_positions,
      decoder_mask=decoder_mask,
      deterministic=not enable_dropout,
      decode=decode,
      max_decode_length=max_decode_length,
      ctx=ctx,
    )
    return logits
