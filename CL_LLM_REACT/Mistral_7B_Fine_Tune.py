import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']="False"
# os.environ[]

import jax
# jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import equinox as eqx
import json
#import sentencepiece as spm
from tokenizer import MistralTokenizer
from model_customized import Transformer
from typing import Optional, Tuple, List, NamedTuple
from jax import random, grad, jit
from sklearn.model_selection import train_test_split
from preprocess_Suzuki_Coupling_data import make_reaction 
from rope import precompute_frequencies
import os
import torch  
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import optax
import nvidia_smi
nvidia_smi.nvmlInit()
i = 0
deviceCount = nvidia_smi.nvmlDeviceGetCount()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print("First -- Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total,\
    info.total/(1024*1024), info.free/(1024*1024), info.used/(1024*1024)))
    
    
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
    max_batch_size: int = 16


#Load the pre-trained Mistral model, tokenize inputs
class Mistral7B:
    def __init__(self, mistralpath, key, dtype=jnp.bfloat16):
        self.mistralpath = mistralpath
        self.mistral_param_path = os.path.join(mistralpath, 'params.json')
        self.mistral_pretrained_path = os.path.join(mistralpath, 'mistral7B_jax_port_fast.eqx')
        self.tokenizer_path = os.path.join(mistralpath, 'tokenizer.model')
        self.sp = MistralTokenizer(self.tokenizer_path)
        with open(self.mistral_param_path, "r") as f:
            self.args = ModelArgs(**json.loads(f.read()))
        self.mistral_model = Transformer(self.args, key, dtype)
        self.mistral_model = eqx.tree_deserialise_leaves(self.mistral_pretrained_path , self.mistral_model)
    
    def get_mistral_pretrained(self):
        self.mistral_model = eqx.tree_deserialise_leaves(self.mistral_pretrained_path , self.mistral_model)

    def tokenize_smiles(self, smiles):
        return self.sp.encode(smiles)

    def _precompute(self,max_len):
        
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("cache k --Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total,\
            info.total/(1024*1024), info.free/(1024*1024), info.used/(1024*1024)))
        
        cache_k = jnp.zeros((self.args.max_batch_size, self.args.n_layers, self.args.sliding_window, self.args.n_kv_heads, self.args.head_dim), dtype=jnp.bfloat16)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("cache v --Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total,\
            info.total/(1024*1024), info.free/(1024*1024), info.used/(1024*1024)))
        cache_v = jnp.zeros((self.args.max_batch_size, self.args.n_layers, self.args.sliding_window, self.args.n_kv_heads, self.args.head_dim), dtype=jnp.bfloat16)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("cos_freq, sin_freq -- Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total,\
            info.total/(1024*1024), info.free/(1024*1024), info.used/(1024*1024)))
        cos_freq, sin_freq = precompute_frequencies(self.args.head_dim, max_len)
        positions = jnp.arange(0, max_len)
        positions_padded = jnp.pad( positions, (0, self.args.sliding_window - len(positions)), constant_values=-1)
        
        
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("After the model is defined --Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total,\
            info.total/(1024*1024), info.free/(1024*1024), info.used/(1024*1024)))


        return cache_k, cache_v, cos_freq, sin_freq, positions_padded


# Define the regression block with a Multi-head attention layer
class MultiHeadAttentionRegression(eqx.Module):
    mha: eqx.nn.MultiheadAttention
    ffn: eqx.nn.Sequential
    rmsnorm1: eqx.nn.RMSNorm
    rmsnorm2: eqx.nn.RMSNorm
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    output_layer: eqx.nn.Linear

    def __init__(self, num_heads, embed_dim, key):

        assert embed_dim > 0, "embed_dim must be a positive integer"
        
        # Split key for reproducibility
        subkey_mha, subkey_ffn, subkey_linear1, subkey_linear2, subkey_output = jax.random.split(key, 5)

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
        self.linear1 = eqx.nn.Linear(embed_dim, 256, key=subkey_linear1, dtype=jnp.float32)
        self.linear2 = eqx.nn.Linear(256, 128, key=subkey_linear2, dtype=jnp.float32)
        self.output_layer = eqx.nn.Linear(128, 1, key=subkey_output,  dtype=jnp.float32)

    def __call__(self, x):
        #x = mistral(x)
        x =  x + self.mha(x, x, x)  # Residual connection
        x = jax.vmap(self.rmsnorm1)(x)
        ff = jax.vmap(self.ffn)(x)
        x = x + ff
        x = jax.vmap(self.rmsnorm2)(x)
        x = jax.vmap(self.linear1)(x)
        x = jax.nn.silu(x)
        x = jax.vmap(self.linear2)(x)
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

#Test these classes below!
key=jax.random.PRNGKey(0)
path="/vast/users/kraghavan/Mistral/"


info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print("Before the model is defined -- Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle),\
    100*info.free/info.total, info.total/(1024*1024), info.free/(1024*1024), info.used/(1024*1024)))

Mistral = Mistral7B(path,key) #<---- Class and laod model weights internally
# Mistral.get_mistral_pretrained() #<---- Actual model weights!

info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print("After the model is defined --Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total,\
    info.total/(1024*1024), info.free/(1024*1024), info.used/(1024*1024)))

# Data Loading and Preprocessing
data_file  = "../data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx"
df = make_reaction(data_file) #df[['rxn', 'y']]
#tokenized_smiles, yields, max_len = preprocess_data(df)

tokenized_rxn = [Mistral.tokenize_smiles(smiles) for smiles in df['rxn']]
max_len =     max(len(sublist) for sublist in tokenized_rxn) #len(tokenized_rxn[0])
padded_a = [sublist + [0] * (max_len - len(sublist)) for sublist in tokenized_rxn]
inp_array = np.stack([np.array(aa) for aa in padded_a]) # jnp.array(padded_a)
yields = np.array(df['y'], dtype=jnp.float32)


info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print("Before precompute -- Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle),\
    100*info.free/info.total, info.total/(1024*1024), info.free/(1024*1024), info.used/(1024*1024)))

cache_k, cache_v, cos_freq, sin_freq, positions_padded = Mistral._precompute(max_len)

info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print("After precompute -- Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

# 80/20 split
train_rxn, val_rxn, train_yields, val_yields = train_test_split(
    inp_array, yields, test_size=0.2, random_state=42
)

print(train_rxn.shape, type(train_rxn),  val_rxn.shape)
# Convert JAX arrays to PyTorch tensors
train_rxns_torch = torch.tensor(train_rxn)
train_yields_torch = torch.tensor(train_yields, dtype=torch.float32)
val_rxns_torch = torch.tensor(val_rxn)
val_yields_torch = torch.tensor(val_yields, dtype=torch.float32)

# Create dataset objects for training and validation
train_dataset = ReactionDataset(train_rxns_torch, train_yields_torch)
val_dataset = ReactionDataset(val_rxns_torch, val_yields_torch)

print("\n Batch size:",Mistral.args.max_batch_size)
train_loader = DataLoader(train_dataset, batch_size=Mistral.args.max_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Mistral.args.max_batch_size, shuffle=False)

# Initialize the regression block
embed_dim = 4096  # Assuming embedding dimension of Mistral
num_heads = 1  # Single attention head for regression task
predictor = MultiHeadAttentionRegression(num_heads, embed_dim,  key)

info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))


# Step 1: Fine-tune the model
@eqx.filter_value_and_grad
def loss_fn(predictor, model,  batch_rxns, batch_yields, cache_k, cache_v, cos_freq, sin_freq, positions_padded ):

    batch_rxns = jnp.array(batch_rxns.numpy())
    batch_yields = jnp.array(batch_yields.numpy())
   
    model = jax.vmap(model, in_axes= (0, None, None, None, None, 0, 0))
    embeddings, cache_k, cache_v = model(batch_rxns, cos_freq, sin_freq, positions_padded, None, cache_k, cache_v)
    print("\nEmbedding shape: ",embeddings.shape)
    predictions = jax.vmap(predictor)(embeddings)

    loss = jnp.mean((predictions - batch_yields) ** 2)
    return loss

#Step 2 : Training step 
@eqx.filter_jit
def train_step(predictor, model, batch_rxns, batch_yields, cache_k, cache_v, cos_freq, sin_freq, positions_padded, opt_state):
    # Get loss and gradients
    loss, grads = loss_fn(predictor, model, batch_rxns, batch_yields, cache_k, cache_v, cos_freq, sin_freq, positions_padded)
    # Update the parameters using the optimizer
    updates, opt_state = optimizer.update(grads, opt_state, predictor)
    predictor = eqx.apply_updates(predictor, updates)
    
    return loss, predictor, opt_state

# Step 3: Initialize optimizer
optimizer = optax.adamw(learning_rate=1e-4)
opt_state = optimizer.init(eqx.filter(predictor, eqx.is_array))  # optimizer.init(predictor)
num_epochs = 2
step = 1 
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))


# Training Loop with DataLoader
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_rxn, batch_yields in train_loader:
        # Perform a training step
        loss, predictor, opt_state  = train_step( predictor, Mistral.mistral_model,  batch_rxn, batch_yields, cache_k, cache_v, cos_freq, sin_freq, positions_padded, opt_state)
        
        loss = loss.item()
        print(f"step={epoch}, loss={loss}")
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total/(1024*1024*1024), info.free/(1024*1024*1024), info.used/(1024*1024*1024)))

        # Accumulate loss for monitoring
        running_loss += loss
       
    # Print epoch loss
    print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader)}")
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total/(1024*1024*1024), info.free/(1024*1024*1024), info.used/(1024*1024*1024)))

