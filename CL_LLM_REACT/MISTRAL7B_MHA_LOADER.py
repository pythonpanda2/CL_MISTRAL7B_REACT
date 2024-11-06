"""
@author: pythonpanda2 (aka Ganesh Sivaraman)
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import json
from tokenizer import MistralTokenizer
#from model import Transformer
from model_regular_inteference import Transformer
from typing import Optional, Tuple, List, NamedTuple
import matplotlib.pyplot as plt
from preprocess_Suzuki_Coupling_data import make_reaction 
from rope import precompute_frequencies
import os
from torch.utils.data import Dataset, DataLoader
import quax

# Mistral-7B Model setting
class ModelArgs(NamedTuple):
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int
    vocab_size: int
    sliding_window: int
    norm_eps: float
    max_batch_size: int = 2


#Load the pre-trained Mistral model, tokenize inputs
class Mistral7B:
    def __init__(self, mistralpath, key, dtype=jnp.bfloat16):
        self.mistralpath = mistralpath
        self.mistral_param_path = os.path.join(mistralpath, 'params.json')
        self.mistral_pretrained_path = os.path.join(mistralpath, 'mistral7B_jax_port_regular.eqx')  #  os.path.join(mistralpath, 'mistral7B_jax_port_fast.eqx')
        self.tokenizer_path = os.path.join(mistralpath, 'tokenizer.model')
        self.sp = MistralTokenizer(self.tokenizer_path)
        with open(self.mistral_param_path, "r") as f:
            self.args = ModelArgs(**json.loads(f.read()))
        self.mistral_model = Transformer(self.args, key, dtype)
        self.mistral_model = eqx.tree_deserialise_leaves(self.mistral_pretrained_path , self.mistral_model)
        #self.mistral_model = quax.lora.loraify(self.mistral_model, rank=8, key=key) 
    #def get_mistral_pretrained(self):
        #return eqx.tree_deserialise_leaves(self.mistral_pretrained_path , self.mistral_model)

    def tokenize_smiles(self, smiles):
        return self.sp.encode(smiles)

    def _precompute(self,max_len):
        cache_k = jnp.zeros((self.args.max_batch_size, self.args.n_layers, self.args.sliding_window, self.args.n_kv_heads, self.args.head_dim), dtype=jnp.bfloat16)
        cache_v = jnp.zeros((self.args.max_batch_size, self.args.n_layers, self.args.sliding_window, self.args.n_kv_heads, self.args.head_dim), dtype=jnp.bfloat16)
        cos_freq, sin_freq = precompute_frequencies(self.args.head_dim, max_len)
        positions = jnp.arange(0, max_len)
        positions_padded = jnp.pad( positions, (0, self.args.sliding_window - len(positions)), constant_values=-1)
        return  cos_freq, sin_freq, positions_padded, cache_k, cache_v


# Define the regression block with a Multi-head attention layer
class MultiHeadAttentionRegression(eqx.Module):
    rope_embeddings: eqx.nn.RotaryPositionalEmbedding
    mha: eqx.nn.MultiheadAttention
    ffn: eqx.nn.Sequential
    rmsnorm1: eqx.nn.RMSNorm
    rmsnorm2: eqx.nn.RMSNorm
    rmsnorm3: eqx.nn.RMSNorm
    rmsnorm4: eqx.nn.RMSNorm
    dropout: eqx.nn.Dropout
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    output_layer: eqx.nn.Linear

    def __init__(self, num_heads, embed_dim, key):

        assert embed_dim > 0, "embed_dim must be a positive integer"
        
        # Split key for reproducibility
        subkey_mha, subkey_ffn,  subkey_linear1, subkey_linear2, subkey_output = jax.random.split(key, 5)
        
        self.rope_embeddings = eqx.nn.RotaryPositionalEmbedding(embedding_size=embed_dim // num_heads, dtype=jnp.float32)
        self.mha = eqx.nn.MultiheadAttention(num_heads=num_heads, query_size=embed_dim, dtype=jnp.float32,  key=subkey_mha)
        self.ffn = eqx.nn.MLP(
            embed_dim,
            out_size=embed_dim,
            width_size= embed_dim,
            depth=2,
            activation = jax.nn.silu,
            key=subkey_ffn,
        )
        self.rmsnorm1 = eqx.nn.RMSNorm(shape=embed_dim,  dtype=jnp.float32)
        self.rmsnorm2 = eqx.nn.RMSNorm(shape=embed_dim,  dtype=jnp.float32)
        self.dropout = eqx.nn.Dropout(p=0.2)
        self.linear1 = eqx.nn.Linear(embed_dim, 1024, key=subkey_linear1, dtype=jnp.float32)
        self.rmsnorm3 = eqx.nn.RMSNorm(shape=1024,  dtype=jnp.float32)
        self.linear2 = eqx.nn.Linear(1024, 256, key=subkey_linear2, dtype=jnp.float32)
        self.rmsnorm4 = eqx.nn.RMSNorm(shape=256,  dtype=jnp.float32)
        self.output_layer = eqx.nn.Linear(256, 1, key=subkey_output,  dtype=jnp.float32)

    def __call__(self, x, key=None, is_training=True):
        #x = mistral(x)
        
        if is_training:
            if key is None:
                raise ValueError("Dropout requires a key when running in training mode.")
            # Split key to use different keys for different dropout calls
            subkey_dropout_mha, subkey_dropout1, subkey_dropout2, subkey_dropout3 = jax.random.split(key, 4)

        def process_heads(
            query_heads: jnp.ndarray,
            key_heads: jnp.ndarray,
            value_heads: jnp.ndarray
        ) -> tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray
        ]:
            query_heads = jax.vmap(self.rope_embeddings,
                                   in_axes=1,
                                   out_axes=1)(query_heads)
            key_heads = jax.vmap(self.rope_embeddings,
                                 in_axes=1,
                                 out_axes=1)(key_heads)

            return query_heads, key_heads, value_heads

        mha_out = self.mha(x, x, x, process_heads=process_heads)
        if is_training:
            mha = self.dropout(mha_out, key=subkey_dropout_mha)
        x =  x + mha_out  # Residual connection

        x = jax.vmap(self.rmsnorm1)(x) #Layer norm

        ff = jax.vmap(self.ffn)(x) #MLP
        if is_training:
            ff = self.dropout(ff, key=subkey_dropout1)
        x = x + ff   #Residual
        x = jax.vmap(self.rmsnorm2)(x)  #Layer norm

        #(Optional : Drop out after-->) Linear-->  BatchNorm-->  Swish blocks
        x = jax.vmap(self.linear1)(x) 
        if is_training:
            x = self.dropout(x, key=subkey_dropout2)
        x = jax.vmap(self.rmsnorm3)(x)
        x = jax.nn.silu(x)

        x = jax.vmap(self.linear2)(x) 
        if is_training:
            x = self.dropout(x, key=subkey_dropout3)
        x = jax.vmap(self.rmsnorm4)(x)
        x = jax.nn.silu(x)

        return jax.vmap(self.output_layer)(x)  # Output a scalar value

class ReactionDataset(Dataset):
    def __init__(self, rxn, yields):
        self.rxn = rxn
        self.yields = yields

    def __len__(self):
        return len(self.rxn)

    def __getitem__(self, idx):
        return self.rxn[idx], self.yields[idx]

