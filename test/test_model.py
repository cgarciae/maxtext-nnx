import dataclasses
from typing import Any
from MaxText import layers, layers_nnx, pyconfig
import yaml
import nnx
import jax
import jax.numpy as jnp

class Box:
    def __init__(self, dict_: dict[str, Any]):
        self._dict = dict_

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
params = variables["params"]

# init nnx
module_nnx = layers_nnx.Transformer(config, ctx=nnx.context(0))

print(module_flax)