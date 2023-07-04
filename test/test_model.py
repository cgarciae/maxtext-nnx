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
flax_params = variables["params"]
flax_flat_params = traverse_util.flatten_dict(flax_params, sep="/")

# init nnx
module_nnx = layers_nnx.Transformer(config, ctx=nnx.context(0))
nnx_params = module_nnx.filter("params")

assert len(flax_flat_params) == len(nnx_params)

assert flax_flat_params["token_embedder/embedding"].value.shape == nnx_params["decoder/shared_embedding/embedding"].value.shape
assert flax_flat_params["token_embedder/embedding"].names == nnx_params["decoder/shared_embedding/embedding"].sharding
module_nnx.token_embedder.embedding = flax_flat_params["token_embedder/embedding"].value

assert flax_flat_params["decoder/decoder/mlp/wi/kernel"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/mlp/wi/kernel"].value.shape
assert flax_flat_params["decoder/decoder/mlp/wi/kernel"].names == nnx_params["decoder/layers/scan_module/remat_module/mlp/wi/kernel"].sharding
module_nnx.decoder.layers.scan_module.remat_module.mlp.wi.kernel = flax_flat_params["decoder/decoder/mlp/wi/kernel"].value

assert flax_flat_params["decoder/decoder/mlp/wo/kernel"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/mlp/wo/kernel"].value.shape
assert flax_flat_params["decoder/decoder/mlp/wo/kernel"].names == nnx_params["decoder/layers/scan_module/remat_module/mlp/wo/kernel"].sharding
module_nnx.decoder.layers.scan_module.remat_module.mlp.wo.kernel = flax_flat_params["decoder/decoder/mlp/wo/kernel"].value

assert flax_flat_params["decoder/decoder/pre_self_attention_layer_norm/scale"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/pre_self_attention_layer_norm/scale"].value.shape
assert flax_flat_params["decoder/decoder/pre_self_attention_layer_norm/scale"].names == nnx_params["decoder/layers/scan_module/remat_module/pre_self_attention_layer_norm/scale"].sharding
module_nnx.decoder.layers.scan_module.remat_module.pre_self_attention_layer_norm.scale = flax_flat_params["decoder/decoder/pre_self_attention_layer_norm/scale"].value

assert flax_flat_params["decoder/decoder/relpos_bias/rel_embedding"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/relpos_bias/rel_embedding"].value.shape
assert flax_flat_params["decoder/decoder/relpos_bias/rel_embedding"].names == nnx_params["decoder/layers/scan_module/remat_module/relpos_bias/rel_embedding"].sharding
module_nnx.decoder.layers.scan_module.remat_module.relpos_bias.rel_embedding = flax_flat_params["decoder/decoder/relpos_bias/rel_embedding"].value

assert flax_flat_params["decoder/decoder/self_attention/key/kernel"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/self_attention/key/kernel"].value.shape
assert flax_flat_params["decoder/decoder/self_attention/key/kernel"].names == nnx_params["decoder/layers/scan_module/remat_module/self_attention/key/kernel"].sharding
module_nnx.decoder.layers.scan_module.remat_module.self_attention.key.kernel = flax_flat_params["decoder/decoder/self_attention/key/kernel"].value

assert flax_flat_params["decoder/decoder/self_attention/key_layer_norm/scale"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/self_attention/key_layer_norm/scale"].value.shape
assert flax_flat_params["decoder/decoder/self_attention/key_layer_norm/scale"].names == nnx_params["decoder/layers/scan_module/remat_module/self_attention/key_layer_norm/scale"].sharding
module_nnx.decoder.layers.scan_module.remat_module.self_attention.key_layer_norm.scale = flax_flat_params["decoder/decoder/self_attention/key_layer_norm/scale"].value

assert flax_flat_params["decoder/decoder/self_attention/out/kernel"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/self_attention/out/kernel"].value.shape
assert flax_flat_params["decoder/decoder/self_attention/out/kernel"].names == nnx_params["decoder/layers/scan_module/remat_module/self_attention/out/kernel"].sharding
module_nnx.decoder.layers.scan_module.remat_module.self_attention.out.kernel = flax_flat_params["decoder/decoder/self_attention/out/kernel"].value

assert flax_flat_params["decoder/decoder/self_attention/query/kernel"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/self_attention/query/kernel"].value.shape
assert flax_flat_params["decoder/decoder/self_attention/query/kernel"].names == nnx_params["decoder/layers/scan_module/remat_module/self_attention/query/kernel"].sharding
module_nnx.decoder.layers.scan_module.remat_module.self_attention.query.kernel = flax_flat_params["decoder/decoder/self_attention/query/kernel"].value

assert flax_flat_params["decoder/decoder/self_attention/query_layer_norm/scale"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/self_attention/query_layer_norm/scale"].value.shape
assert flax_flat_params["decoder/decoder/self_attention/query_layer_norm/scale"].names == nnx_params["decoder/layers/scan_module/remat_module/self_attention/query_layer_norm/scale"].sharding
module_nnx.decoder.layers.scan_module.remat_module.self_attention.query_layer_norm.scale = flax_flat_params["decoder/decoder/self_attention/query_layer_norm/scale"].value

assert flax_flat_params["decoder/decoder/self_attention/value/kernel"].value.shape == nnx_params["decoder/layers/scan_module/remat_module/self_attention/value/kernel"].value.shape
assert flax_flat_params["decoder/decoder/self_attention/value/kernel"].names == nnx_params["decoder/layers/scan_module/remat_module/self_attention/value/kernel"].sharding
module_nnx.decoder.layers.scan_module.remat_module.self_attention.value.kernel = flax_flat_params["decoder/decoder/self_attention/value/kernel"].value

assert flax_flat_params["decoder/decoder_norm/scale"].value.shape == nnx_params["decoder/decoder_norm/scale"].value.shape
assert flax_flat_params["decoder/decoder_norm/scale"].names == nnx_params["decoder/decoder_norm/scale"].sharding
module_nnx.decoder.decoder_norm.scale = flax_flat_params["decoder/decoder_norm/scale"].value

y_flax = module_flax.apply(
    {"params": flax_params},
    decoder_input_tokens=jnp.arange(10)[None].astype(dtype=jnp.int32),
    decoder_target_tokens=jnp.arange(10)[None].astype(dtype=jnp.int32),
)
y_nnx = module_nnx(
    decoder_input_tokens=jnp.arange(10)[None].astype(dtype=jnp.int32),
    decoder_target_tokens=jnp.arange(10)[None].astype(dtype=jnp.int32),
    ctx=nnx.context(0),
)

np.testing.assert_allclose(y_flax, y_nnx, rtol=1e-3, atol=1e-3)
