import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import nnx
import numpy as np
import yaml
from flax import traverse_util

from MaxText import layers, layers_nnx, pyconfig

class Box:
    def __init__(self, dict_: dict[str, Any]):
        vars(self)["_dict"] = dict_

    def __getattr__(self, name: str) -> Any:
        value = self._dict[name]
        if isinstance(value, dict):
            return Box(value)
        return value
    
    def __getitem__(self, name: str) -> Any:
        return self.__getattr__(name)
    
    def __setitem__(self, name: str, value: Any) -> None:
        if isinstance(value, Box):
            value = value._dict
        self._dict[name] = value

    def __setattr__(self, name: str, value: Any) -> None:
        self.__setitem__(name, value)

    def __iter__(self):
        return iter(self._dict)
    
    def __len__(self):
        return len(self._dict)

# init config
config = Box(yaml.safe_load(open("MaxText/configs/base.yml", "r")))
config["logical_axis_rules"] = pyconfig._lists_to_tuples(config["logical_axis_rules"])
config["data_sharding"] = pyconfig._lists_to_tuples(config["data_sharding"])
config['emb_dim'] = config['scale'] * config['base_emb_dim']
config['num_heads'] = config['scale'] * config['base_num_heads']
config['mlp_dim'] = config['scale'] * config['base_mlp_dim']
config['num_decoder_layers'] = config['scale'] * config['base_num_decoder_layers']

# init flax
module_flax = layers.Transformer(config)
variables = module_flax.init(
    jax.random.PRNGKey(0),
    decoder_input_tokens=jnp.ones((1, 2), dtype=jnp.int32),
    decoder_target_tokens=jnp.ones((1, 2), dtype=jnp.int32),
)
param_flax = variables["params"]
flat_params_flax = traverse_util.flatten_dict(param_flax, sep="/")
del variables

# init nnx
module_nnx = layers_nnx.Transformer(config, ctx=nnx.context(0))
nnx_params = module_nnx.filter("params")

assert len(flat_params_flax) == len(nnx_params)

assert flat_params_flax["token_embedder/embedding"].value.shape == nnx_params["decoder/shared_embedding/embedding"].value.shape
assert flat_params_flax["token_embedder/embedding"].names == nnx_params["decoder/shared_embedding/embedding"].sharding
module_nnx.token_embedder.embedding = flat_params_flax["token_embedder/embedding"].value

assert flat_params_flax["decoder/decoder/mlp/wi/kernel"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/mlp/wi/kernel"].value.shape
assert flat_params_flax["decoder/decoder/mlp/wi/kernel"].names == nnx_params["decoder/layers/scan_module/remat_module/mlp/wi/kernel"].sharding
module_nnx.decoder.layers.scan_module.remat_module.mlp.wi.kernel = flat_params_flax["decoder/decoder/mlp/wi/kernel"].value

assert flat_params_flax["decoder/decoder/mlp/wo/kernel"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/mlp/wo/kernel"].value.shape
assert flat_params_flax["decoder/decoder/mlp/wo/kernel"].names == nnx_params["decoder/layers/scan_module/remat_module/mlp/wo/kernel"].sharding
module_nnx.decoder.layers.scan_module.remat_module.mlp.wo.kernel = flat_params_flax["decoder/decoder/mlp/wo/kernel"].value

assert flat_params_flax["decoder/decoder/pre_self_attention_layer_norm/scale"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/pre_self_attention_layer_norm/scale"].value.shape
assert flat_params_flax["decoder/decoder/pre_self_attention_layer_norm/scale"].names == nnx_params["decoder/layers/scan_module/remat_module/pre_self_attention_layer_norm/scale"].sharding
module_nnx.decoder.layers.scan_module.remat_module.pre_self_attention_layer_norm.scale = flat_params_flax["decoder/decoder/pre_self_attention_layer_norm/scale"].value

assert flat_params_flax["decoder/decoder/relpos_bias/rel_embedding"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/relpos_bias/rel_embedding"].value.shape
assert flat_params_flax["decoder/decoder/relpos_bias/rel_embedding"].names == nnx_params["decoder/layers/scan_module/remat_module/relpos_bias/rel_embedding"].sharding
module_nnx.decoder.layers.scan_module.remat_module.relpos_bias.rel_embedding = flat_params_flax["decoder/decoder/relpos_bias/rel_embedding"].value

assert flat_params_flax["decoder/decoder/self_attention/key/kernel"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/self_attention/key/kernel"].value.shape
assert flat_params_flax["decoder/decoder/self_attention/key/kernel"].names == nnx_params["decoder/layers/scan_module/remat_module/self_attention/key/kernel"].sharding
module_nnx.decoder.layers.scan_module.remat_module.self_attention.key.kernel = flat_params_flax["decoder/decoder/self_attention/key/kernel"].value

assert flat_params_flax["decoder/decoder/self_attention/key_layer_norm/scale"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/self_attention/key_layer_norm/scale"].value.shape
assert flat_params_flax["decoder/decoder/self_attention/key_layer_norm/scale"].names == nnx_params["decoder/layers/scan_module/remat_module/self_attention/key_layer_norm/scale"].sharding
module_nnx.decoder.layers.scan_module.remat_module.self_attention.key_layer_norm.scale = flat_params_flax["decoder/decoder/self_attention/key_layer_norm/scale"].value

assert flat_params_flax["decoder/decoder/self_attention/out/kernel"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/self_attention/out/kernel"].value.shape
assert flat_params_flax["decoder/decoder/self_attention/out/kernel"].names == nnx_params["decoder/layers/scan_module/remat_module/self_attention/out/kernel"].sharding
module_nnx.decoder.layers.scan_module.remat_module.self_attention.out.kernel = flat_params_flax["decoder/decoder/self_attention/out/kernel"].value

assert flat_params_flax["decoder/decoder/self_attention/query/kernel"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/self_attention/query/kernel"].value.shape
assert flat_params_flax["decoder/decoder/self_attention/query/kernel"].names == nnx_params["decoder/layers/scan_module/remat_module/self_attention/query/kernel"].sharding
module_nnx.decoder.layers.scan_module.remat_module.self_attention.query.kernel = flat_params_flax["decoder/decoder/self_attention/query/kernel"].value

assert flat_params_flax["decoder/decoder/self_attention/query_layer_norm/scale"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/self_attention/query_layer_norm/scale"].value.shape
assert flat_params_flax["decoder/decoder/self_attention/query_layer_norm/scale"].names == nnx_params["decoder/layers/scan_module/remat_module/self_attention/query_layer_norm/scale"].sharding
module_nnx.decoder.layers.scan_module.remat_module.self_attention.query_layer_norm.scale = flat_params_flax["decoder/decoder/self_attention/query_layer_norm/scale"].value

assert flat_params_flax["decoder/decoder/self_attention/value/kernel"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/self_attention/value/kernel"].value.shape
assert flat_params_flax["decoder/decoder/self_attention/value/kernel"].names == nnx_params["decoder/layers/scan_module/remat_module/self_attention/value/kernel"].sharding
module_nnx.decoder.layers.scan_module.remat_module.self_attention.value.kernel = flat_params_flax["decoder/decoder/self_attention/value/kernel"].value

assert flat_params_flax["decoder/decoder_norm/scale"].value.shape == nnx_params["decoder/decoder_norm/scale"].value.shape
assert flat_params_flax["decoder/decoder_norm/scale"].names == nnx_params["decoder/decoder_norm/scale"].sharding
module_nnx.decoder.decoder_norm.scale = flat_params_flax["decoder/decoder_norm/scale"].value

y_flax = module_flax.apply(
    {"params": param_flax},
    decoder_input_tokens=jnp.arange(10)[None].astype(dtype=jnp.int32),
    decoder_target_tokens=jnp.arange(10)[None].astype(dtype=jnp.int32),
)
y_nnx = module_nnx(
    decoder_input_tokens=jnp.arange(10)[None].astype(dtype=jnp.int32),
    decoder_target_tokens=jnp.arange(10)[None].astype(dtype=jnp.int32),
    ctx=nnx.context(0),
)

np.testing.assert_allclose(y_flax, y_nnx, rtol=1e-3, atol=1e-3)

#------------------
# test encoding
#------------------
seq_len = 10

# init cache flax
variables = module_flax.init(
    jax.random.PRNGKey(0),
    decoder_input_tokens=jnp.ones((1, seq_len), dtype=jnp.int32),
    decoder_target_tokens=jnp.ones((1, seq_len), dtype=jnp.int32),
    decode=True,
)
cache_flax = variables["cache"]
flat_cache_flax = traverse_util.flatten_dict(cache_flax, sep="/")
del variables

# init cache nnx
module_nnx.for_each(
    layers_nnx.MultiHeadDotProductAttention, 
    lambda x: x.init_cache(1, seq_len)
)
cache_nnx = module_nnx.filter("cache")

# flax -> nnx
assert flat_cache_flax["decoder/decoder/self_attention/cache_index"].shape == cache_nnx["decoder/layers/scan_module/remat_module/self_attention/cache_index"].value.shape
module_nnx.decoder.layers.scan_module.remat_module.self_attention.cache_index = flat_cache_flax["decoder/decoder/self_attention/cache_index"]

assert flat_cache_flax["decoder/decoder/self_attention/cached_key"].shape == cache_nnx["decoder/layers/scan_module/remat_module/self_attention/cached_key"].value.shape
module_nnx.decoder.layers.scan_module.remat_module.self_attention.cached_key = flat_cache_flax["decoder/decoder/self_attention/cached_key"]

assert flat_cache_flax["decoder/decoder/self_attention/cached_value"].shape == cache_nnx["decoder/layers/scan_module/remat_module/self_attention/cached_value"].value.shape
module_nnx.decoder.layers.scan_module.remat_module.self_attention.cached_value = flat_cache_flax["decoder/decoder/self_attention/cached_value"]


# inference flax
y_flax, updates = module_flax.apply(
    {"params": param_flax, "cache": cache_flax},
    decoder_input_tokens=jnp.arange(1)[None].astype(dtype=jnp.int32),
    decoder_target_tokens=jnp.arange(1)[None].astype(dtype=jnp.int32),
    decode=True,
    mutable=["cache"],
)
cache_flax = updates["cache"]
flat_cache_flax = traverse_util.flatten_dict(cache_flax, sep="/")
del updates

# inference nnx
y_nnx = module_nnx(
    decoder_input_tokens=jnp.arange(1)[None].astype(dtype=jnp.int32),
    decoder_target_tokens=jnp.arange(1)[None].astype(dtype=jnp.int32),
    decode=True,
    ctx=nnx.context(0),
)
cache_nnx = module_nnx.filter("cache")

np.testing.assert_allclose(y_flax, y_nnx, rtol=1e-3, atol=1e-3)

# np.testing.assert_allclose refuses to work for bfloat16
# so we are calculating the difference manually
assert np.sqrt(
    (
        (
            flat_cache_flax["decoder/decoder/self_attention/cache_index"] -
            cache_nnx["decoder/layers/scan_module/remat_module/self_attention/cache_index"].value
        ) ** 2
    ).sum()
) < 1e-3
assert np.sqrt(
    (
        (
            flat_cache_flax["decoder/decoder/self_attention/cached_key"] -
            cache_nnx["decoder/layers/scan_module/remat_module/self_attention/cached_key"].value
        ) ** 2
    ).sum()
) < 1e-3
assert np.sqrt(
    (
        (
            flat_cache_flax["decoder/decoder/self_attention/cached_value"] -
            cache_nnx["decoder/layers/scan_module/remat_module/self_attention/cached_value"].value
        ) ** 2
    ).sum()
) < 1e-3

cache_flax