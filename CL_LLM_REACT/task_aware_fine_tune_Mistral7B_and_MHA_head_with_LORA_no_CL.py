"""
@author: pythonpanda2 (aka Ganesh Sivaraman)
""" 
from MISTRAL7B_MHA_LOADER import * 
import argparse 
import functools as ft
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import equinox as eqx
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from preprocess_Suzuki_Coupling_data import create_task_aware_reaction_df, task_aware_splits 
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd 
import optax
import quax

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".70"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
#device = torch.device("cpu")


def get_argparse():
    """
    A function to parse all the input arguments
    """
    parser = argparse.ArgumentParser(description='A python workflow to run Low-Rank Adaptation (LoRA) fine tuning on pre-trained  Mistral7B + MHA jointly')
    parser.add_argument('-p','--path', type=str,required=True,help='Full path to model file')
    parser.add_argument('-xl','--xlsfile', type=str,required=True,help='Full path to the reaction excel format file containing the reaction data')
    parser.add_argument('-N','--epoch',type=int,metavar='',\
                       help="Number of epochs to fine tune the MHA head",default=2)
    parser.add_argument('-rs','--randomseed',type=int,metavar='',\
            help="Initialize the random seed",default=0)
    parser.add_argument('-r','--rank',type=int,metavar='',\
            help="LoRA Rank",default=8)
    parser.add_argument('-s','--scale',type=float,metavar='',\
            help="LoRA Scale: [1.] scaling = alpha / r [2.] weight += (lora_B @ lora_A) * scaling",default=0.01)
    parser.add_argument('-lr','--learning_rate',type=float,metavar='',\
            help="Optax learning rate",default=1e-5)
    parser.add_argument('-nh','--num_heads',type=int,metavar='',\
            help="Sepecify the number of heas for the multi-head attention based regression block",default=4)
    return parser.parse_args()

@eqx.filter_jit
def compute_val(params, static,  val_rxns,  cos_freq, sin_freq, positions_padded, cache_k, cache_v ):
    predictor = eqx.combine(params, static)
    # Pass is_training=False during validation/inference to deactivate dropout and also set drop out key to None. 
    predictions = predictor( val_rxns,  cos_freq, sin_freq, positions_padded, cache_k, cache_v, None, False)

    return predictions

# Define the plotting function
def plot_true_vs_predicted(true_labels, predicted_labels, rmse, r2):
    plt.figure(figsize=(10, 6))
    plt.scatter(true_labels, predicted_labels, color='b', alpha=0.6, s=10, label='Predicted vs True')
    plt.plot([true_labels.min(), true_labels.max()], [true_labels.min(), true_labels.max()], 'k--', lw=2)
    plt.xlabel('True Yield', fontsize=12)
    plt.ylabel('Predicted Yield', fontsize=12)
    plt.title('True vs Predicted Yield', fontsize=14)
    plt.legend([f'RMSE: {rmse:.2f}\nR²: {r2:.2f}'], loc='upper left')
    plt.grid(True)
    plt.savefig("true_vs_pred_yield.png", dpi=500, bbox_inches='tight')


def main():
    '''
    Gather all the workflow here!
    '''
    args = get_argparse()

    #Set up arguments!
    seed = args.randomseed
    np.random.seed(seed)
    print("\nSet random seed :", seed )
    lr = args.learning_rate
    lora_rank = args.rank
    lora_scale = args.scale
    num_epochs = args.epoch

    key=jax.random.key(seed)          # jax.random.PRNGKey(args.randomseed)
    mistral_key, setup_key, mha_dropout_key = jax.random.split(key, 3)
    path = args.path
    #Load Mistral Class!
    Mistral = Mistral7B(path, mistral_key) #<---- Class

    # Data Loading and Preprocessing
    #data_file  = "/eagle/FOUND4CHEM/project/CL_4_RXN/CL_MISTRAL7B_REACT/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx"
    data_file = args.xlsfile
    task_aware_reactions_df  = create_task_aware_reaction_df(data_file) #df[['rxn', 'y']]
    #task_aware_reactions_df = task_aware_splits(df)
    #tokenized_smiles, yields, max_len = preprocess_data(df)

    tokenized_rxn = [Mistral.tokenize_smiles(smiles) for smiles in task_aware_reactions_df['rxn']]
    max_len =     max(len(sublist) for sublist in tokenized_rxn) #len(tokenized_rxn[0])
    print(f"\n Sequence length: {max_len}")
    del tokenized_rxn # We dont need this after the max_len is estimated!
    
    # 70/30 split
    train_df, test_df = train_test_split(task_aware_reactions_df, test_size=0.3, random_state=seed)
    tokenize_test_rxn = [Mistral.tokenize_smiles(smiles) for smiles in test_df['rxn']]
    padded_test_rxn = [sublist + [0] * (max_len - len(sublist)) for sublist in tokenize_test_rxn]
    val_rxn =  np.stack([np.array(aa) for aa in padded_test_rxn])
    val_yields = np.array(test_df['y'], dtype=np.float32)
    # Convert JAX arrays to PyTorch tensors
    #train_rxns_torch = torch.tensor(train_rxn)
    #train_yields_torch = torch.tensor(train_yields, dtype=torch.float32)
    val_rxns_torch = torch.tensor(val_rxn)
    val_yields_torch = torch.tensor(val_yields, dtype=torch.float32)

    # Create dataset objects for training and validation
    #train_dataset = ReactionDataset(train_rxns_torch, train_yields_torch)
    val_dataset = ReactionDataset(val_rxns_torch, val_yields_torch)

    print("\n Batch size:",Mistral.args.max_batch_size)
    #train_loader = DataLoader(train_dataset, batch_size=Mistral.args.max_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Mistral.args.max_batch_size, shuffle=True)

    # Initialize the regression block
    embed_dim = 4096  # Assuming embedding dimension of Mistral
    num_heads = args.num_heads  # Number of  attention head for regression task
    print(f"\n Setting up MHA regression block with {num_heads} heads!")
    print(f"\n Setting up LoRA with a rank of {lora_rank}, and scale of {lora_scale}")

    #Combine the Mistral-7B +  MHA head in to a  unifed model class for fine tuning. 
    class YieldPredictor(eqx.Module):
        model: eqx.Module  # Pre-trained Mistral model
        mha_head: eqx.Module #MHA head 

        def __init__(self, model, num_heads, embed_dim, setup_key, lora_rank, lora_scale):
            lora_key, mha_key = jax.random.split(setup_key, 2)
            self.model =  quax.lora.loraify(model, rank=lora_rank, scale=lora_scale, key=lora_key)
            self.mha_head = SimpleMultiHeadAttentionRegression(num_heads, embed_dim,  mha_key)

        def __call__(self,  batch_rxns,  cos_freq, sin_freq, positions_padded, cache_k, cache_v, dropout_key, is_training):
            quaxified_model = quax.quaxify(self.model)
            remat_model = jax.remat(quaxified_model) 

            vmap_model = jax.vmap(remat_model, in_axes= (0, None, None, None, None, 0, 0))
            embeddings, _, _ = vmap_model(batch_rxns, cos_freq, sin_freq, positions_padded, None, cache_k, cache_v)

            # Pass the embeddings through the regression head
            yield_prediction = jax.vmap(lambda emb: self.mha_head(emb, dropout_key, is_training))(embeddings) # jax.vmap --> eqx.filter_vmap
            return yield_prediction

    
    # model is defined 
    predictor = YieldPredictor(Mistral.mistral_model, num_heads, embed_dim,  setup_key, lora_rank, lora_scale)
    # break the model into trainable parameter and untrainable parameters
    params, static = eqx.partition(predictor, eqx.is_array)
    #print("\nSanity  Check: ", predictor.model)
    print("\n Detected Device: ",jax.devices())
    
    # Step 1: Fine-tune the model
    @eqx.filter_value_and_grad
    @eqx.filter_jit
    def loss_fn(params, static,  batch_rxns, batch_yields, mask,  cos_freq, sin_freq, positions_padded, cache_k, cache_v, dropout_key ) : 
        # Pass is_training=True during training
        predictor = eqx.combine(params, static)
        predictions = predictor(batch_rxns,  cos_freq, sin_freq, positions_padded, cache_k, cache_v, dropout_key, True)
        #loss = jnp.mean((predictions - batch_yields) ** 2)
        loss = jnp.mean( optax.losses.l2_loss(predictions.squeeze(), batch_yields) ) 

        loss = loss * mask  # Zero out losses for padded samples
    
        # Compute mean loss over real samples
        loss = jnp.sum(loss) / jnp.sum(mask)

        return loss

    #Step 2 : Training step
    @eqx.filter_jit
    def train_step(params, static,  batch_rxns, batch_yields, mask, cos_freq, sin_freq, positions_padded, cache_k, cache_v, dropout_key, opt_state):
        # Get loss and gradients
        loss, grads = loss_fn(params, static, batch_rxns, batch_yields, mask,  cos_freq, sin_freq, positions_padded, cache_k, cache_v, dropout_key)
        # Update the parameters using the optimizer
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return loss, params, opt_state

    # Step 3: Initialize optimizer
    print("\n Setting up optimizer with a learning rate = ", lr)
    optimizer = optax.adafactor(learning_rate=lr) #adafactor, lion
    opt_state = optimizer.init(params)  # optimizer.init(predictor)
    
    # Step 4: Run Fine tuning / training
    def pad_collate_fn(batch):
        """ Pad the batches for fixed size!"""
        batch_rxns, batch_yields = zip(*batch)
        batch_size = len(batch_rxns)
        max_batch_size = Mistral.args.max_batch_size  # Fixed batch size (e.g., 12)
    
        # Stack the batch data
        batch_rxns = torch.stack(batch_rxns)  # Shape: (batch_size, seq_length)
        batch_yields = torch.tensor(batch_yields)  # Shape: (batch_size,)
    
        # Create a mask for real samples
        mask = torch.ones(batch_size, dtype=torch.bool)
    
        if batch_size < max_batch_size:
            # Calculate the number of padding samples needed
            num_padding = max_batch_size - batch_size
        
            # Create padding for reactions and yields
            padding_rxn = torch.zeros((num_padding, batch_rxns.shape[1]), dtype=batch_rxns.dtype)
            padding_yield = torch.zeros(num_padding, dtype=batch_yields.dtype)
        
            # Concatenate padding to the batch data
            batch_rxns = torch.cat([batch_rxns, padding_rxn], dim=0)  # Shape: (max_batch_size, seq_length)
            batch_yields = torch.cat([batch_yields, padding_yield], dim=0)  # Shape: (max_batch_size,)
        
            # Extend the mask with False for padded samples
            mask = torch.cat([mask, torch.zeros(num_padding, dtype=torch.bool)], dim=0)  # Shape: (max_batch_size,)
    
        return batch_rxns, batch_yields, mask


    print("\n Total number of training epochs : ", num_epochs )
    task_grouped_train = task_aware_splits(train_df)
    print("\n Total number of training task groups : ",len(task_grouped_train))

    # Training Loop with DataLoader
    task_losses = []  # List to store average loss per task
    for i, key in enumerate(task_grouped_train.keys() ) :
        epoch_losses = []  # List to store epoch losses for this task
        for epoch in range(num_epochs):
            
            task_df = pd.DataFrame(task_grouped_train[key])
            print(f"Task {i+1}: {key}, Samples Size: {task_df.shape[0]}")
            tokenize_task_rxn = [Mistral.tokenize_smiles(smiles) for smiles in task_df['rxn']]
            padded_task_rxn = [sublist + [0] * (max_len - len(sublist)) for sublist in tokenize_task_rxn]
            train_rxn =  np.stack([np.array(aa) for aa in padded_task_rxn ])
            train_yields = np.array(task_df['y'], dtype=np.float32)

            train_rxns_torch = torch.tensor(train_rxn)
            train_yields_torch = torch.tensor(train_yields, dtype=torch.float32)
            train_dataset = ReactionDataset(train_rxns_torch, train_yields_torch)
            train_loader = DataLoader(train_dataset, batch_size=Mistral.args.max_batch_size, shuffle=True, collate_fn=pad_collate_fn)
            step = 1 
            running_loss = 0.0
            print(f"\nTask {i+1}, Epoch {epoch+1}: Total batches = {len(train_loader)}")
            for batch_rxns, batch_yields, mask in train_loader:
                batch_rxns = jnp.array(batch_rxns.numpy())
                batch_yields = jnp.array(batch_yields.numpy(),  dtype=jnp.float32)
                mask = jnp.array(mask.numpy())  # Shape: (max_batch_size,)

                # Reset KV cache for each batch to ensure it's not reused across different batches
                cos_freq, sin_freq, positions_padded, cache_k, cache_v = Mistral._precompute(max_len)
                # Perform a training step
                loss, params, opt_state  = train_step( params, static, batch_rxns, batch_yields, mask, cos_freq, sin_freq, positions_padded, cache_k, cache_v, mha_dropout_key, opt_state)
            
                loss = loss.item()
                #print(f"step={step}, loss={loss}")
                # Accumulate loss for monitoring
                running_loss += loss
                if step % 10 == 0 :
                    avg_loss = running_loss / step
                    print(f"Task {i+1}, Epoch {epoch+1}, Step {step}, Average Loss: {np.round(avg_loss, 6)}")
                step += 1

            # Calculate and record the epoch loss
            epoch_loss =  running_loss/ len(train_loader)
            epoch_losses.append(epoch_loss)
            print(f"Task {i+1}, Epoch {epoch+1} Completed. Epoch Loss: {np.round(epoch_loss, 6)}\n") 

        # Calculate the average loss over all epochs for this task
        avg_task_loss = np.mean(epoch_losses)
        task_losses.append(avg_task_loss)
        print(f"Task {i+1} Completed. Average Loss over {num_epochs} epochs: {np.round(avg_task_loss, 6)}\n")

        # **Save Epoch Losses for the Current Task**
        with open(f"epoch_losses_task_{i+1}.txt", 'w') as f:
            for epoch_idx, loss_value in enumerate(epoch_losses):
                f.write(f"Epoch {epoch_idx+1}: {loss_value}\n")
        print(f"Epoch losses for Task {i+1} saved to 'epoch_losses_task_{i+1}.txt'.")

    # **Save All Task Losses**
    with open("task_losses.txt", 'w') as f:
        for task_idx, loss_value in enumerate(task_losses):
            f.write(f"Task {task_idx+1}: {loss_value}\n")
    print("All task losses saved to 'task_losses.txt'.")

    # Step 5 : After training loop (i.e. validation).
    print("\nTotal number of validation batches : ",len(val_loader))
    running_val_loss = 0.0
    running_r2_score = 0.0
    all_predictions = []
    all_true_labels = []

    step = 1
    for val_rxns, val_yields in val_loader:
        val_rxns = jnp.array(val_rxns.numpy())
        val_yields = jnp.array(val_yields.numpy(), dtype=jnp.float32)

        # Reset KV cache for each batch to ensure it's not reused across different batches
        cos_freq, sin_freq, positions_padded, cache_k, cache_v = Mistral._precompute(max_len)

        # Calculate embeddings and predictions without updating the model
        predictions = compute_val(params, static,  val_rxns, cos_freq, sin_freq, positions_padded, cache_k, cache_v )

        # Calculate validation loss
        val_loss = jnp.mean((predictions.squeeze() - val_yields) ** 2)
        running_val_loss += val_loss.item()
        
        print("\nPredicted shape , values : ", predictions.shape, predictions)
        print("\nTrue Yield : ", val_yields )
        # Collect predictions and true labels
        all_predictions.append(np.asarray(predictions.squeeze(),  dtype=np.float32 ) )  # Flatten predictions
        all_true_labels.append(np.asarray(val_yields,  dtype=np.float32 ))

        # Calculate R² score
        #r2 = r2_score( np.asarray(val_yields), np.asarray( predictions[:, -1, 0]  ) ) # Use just the last dim of pred
        #r2_alt = r2_score( np.asarray(val_yields, dtype=np.float32), np.asarray( jnp.mean(predictions, axis=1).squeeze(),  dtype=np.float32  ) )
        #print("Step , MAE Loss, R2 value for current batch :", step, val_loss, r2_alt)
        #running_r2_score += r2_alt
        step += 1
    print(f"\nValidation Loss: {np.round(running_val_loss / len(train_loader),6 )}")
    # Flatten collected lists to get overall predictions and true labels
    all_predictions = np.concatenate(all_predictions, axis=0,  dtype=np.float32)
    all_true_labels = np.concatenate(all_true_labels, axis=0,  dtype=np.float32)

    # Save true labels and predictions to a .npy file for future use
    np.save("val_true_labels.npy", np.asarray(all_true_labels))
    np.save("val_predictions.npy", np.asarray(all_predictions))

    # Compute RMSE
    rmse = mean_squared_error(all_true_labels, all_predictions, squared=False)
    # Compute R² score
    r2 = r2_score(all_true_labels, all_predictions)
    # Plotting True vs. Predicted
    plot_true_vs_predicted(all_true_labels, all_predictions, rmse, r2)
    # Print RMSE and R²
    print(f"\nValidation RMSE: {np.round(rmse, 5)}")
    print(f"\nValidation R² Score: {np.round(r2, 5)}")

    #val_loss, r2_val=  running_val_loss / len(val_loader),  running_r2_score / len(val_loader)

    #print(f"[Double Check!]Validation Loss: {val_loss}")
    #print(f"[Double Check!] Validation R2: {r2_val}")


# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()
