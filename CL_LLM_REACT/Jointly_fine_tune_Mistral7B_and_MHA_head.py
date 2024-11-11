"""
@author: pythonpanda2 (aka Ganesh Sivaraman)
""" 
from MISTRAL7B_MHA_LOADER import Mistral7B, MultiHeadAttentionRegression, ModelArgs, ReactionDataset    
import argparse 
import jax
import jax.numpy as jnp
import equinox as eqx
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from preprocess_Suzuki_Coupling_data import make_reaction
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
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
    parser = argparse.ArgumentParser(description='A python workflow to run fine tune pre-trained  Mistral7B + MHA jointly')
    parser.add_argument('-p','--path', type=str,required=True,help='Full path to model file')
    parser.add_argument('-xl','--xlsfile', type=str,required=True,help='Full path to the reaction excel format file containing the reaction data')
    parser.add_argument('-N','--epoch',type=int,metavar='',\
                       help="Number of epochs to fine tune the MHA head",default=2)
    parser.add_argument('-rs','--randomseed',type=int,metavar='',\
            help="Initialize the random seed",default=0)
    parser.add_argument('-lr','--learning_rate',type=float,metavar='',\
            help="Optax learning rate",default=1e-5)
    return parser.parse_args()

@eqx.filter_jit
def compute_val(predictor,  val_rxns,  cos_freq, sin_freq, positions_padded, cache_k, cache_v ):

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

    seed = args.randomseed
    np.random.seed(seed)
    lr = args.learning_rate
    print("\nSet random seed :", seed )
    key=jax.random.key(seed)          # jax.random.PRNGKey(args.randomseed)
    model_key, mha_key, mha_dropout_key = jax.random.split(key, 3)
    #path="/eagle/FOUND4CHEM/project/CL_4_RXN/CL_MISTRAL7B_REACT/model_files"
    path = args.path
    Mistral = Mistral7B(path, model_key) #<---- Class
   

    # Data Loading and Preprocessing
    #data_file  = "/eagle/FOUND4CHEM/project/CL_4_RXN/CL_MISTRAL7B_REACT/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx"
    data_file = args.xlsfile
    df = make_reaction(data_file) #df[['rxn', 'y']]
    #tokenized_smiles, yields, max_len = preprocess_data(df)

    tokenized_rxn = [Mistral.tokenize_smiles(smiles) for smiles in df['rxn']]
    max_len =     max(len(sublist) for sublist in tokenized_rxn) #len(tokenized_rxn[0])
    padded_rxn = [sublist + [0] * (max_len - len(sublist)) for sublist in tokenized_rxn]
    tokenized_padded_inp_array = np.stack([np.array(aa) for aa in padded_rxn]) # jnp.array(padded_a)
    yields = np.array(df['y'], dtype=np.float32)

    #cache_k, cache_v, cos_freq, sin_freq, positions_padded = Mistral._precompute(max_len)

    # 80/20 split
    train_rxn, val_rxn, train_yields, val_yields = train_test_split( tokenized_padded_inp_array, yields, test_size=0.2, random_state=seed)

    #print(train_rxn.shape, type(train_rxn),  val_rxn.shape,  type(val_rxn) )
    # Convert JAX arrays to PyTorch tensors
    train_rxns_torch = torch.tensor(train_rxn)
    train_yields_torch = torch.tensor(train_yields, dtype=torch.float32) #float32
    val_rxns_torch = torch.tensor(val_rxn)
    val_yields_torch = torch.tensor(val_yields, dtype=torch.float32) #float32

    # Create dataset objects for training and validation
    train_dataset = ReactionDataset(train_rxns_torch, train_yields_torch)
    val_dataset = ReactionDataset(val_rxns_torch, val_yields_torch)

    print("\n Batch size:",Mistral.args.max_batch_size)
    train_loader = DataLoader(train_dataset, batch_size=Mistral.args.max_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Mistral.args.max_batch_size, shuffle=True)

    # Initialize the regression block
    embed_dim = 4096  # Assuming embedding dimension of Mistral
    num_heads = 4  # Number of  attention head for regression task
    #predictor = MultiHeadAttentionRegression(num_heads, embed_dim,  key)
    
    #Combine the Mistral-7B +  MHA head in to a  unifed model class for fine tuning. 
    class YieldPredictor(eqx.Module):
        model: eqx.Module  # Pre-trained Mistral model
        mha_head: eqx.Module #MHA head 

        def __init__(self, model, num_heads, embed_dim, key):
            self.model = model  # quax.lora.loraify(model, rank=8, key=key)
            self.mha_head = MultiHeadAttentionRegression(num_heads, embed_dim,  key)


        def __call__(self,  batch_rxns,  cos_freq, sin_freq, positions_padded, cache_k, cache_v, key,is_training):
            #remat_model = jax.remat(self.model)
            # Apply remat to each Transformer layer in the Mistral model
            remat_model = jax.remat(self.model)
            vmap_model = jax.vmap(remat_model, in_axes= (0, None, None, None, None, 0, 0)) # jax.vmap --> eqx.filter_vmap
            embeddings, _, _ = vmap_model(batch_rxns, cos_freq, sin_freq, positions_padded, None, cache_k, cache_v)
            # Cast embeddings back to float32 for the predictor
            #embeddings = embeddings.astype(jnp.float32)
            yield_prediction = jax.vmap(lambda emb: self.mha_head(emb, key, is_training))(embeddings) # jax.vmap --> eqx.filter_vmap
            return yield_prediction

    predictor = YieldPredictor(Mistral.mistral_model, num_heads, embed_dim,  mha_key)

    print("\n Detected Device: ",jax.devices())

    # Step 1: Fine-tune the model
    @eqx.filter_value_and_grad
    @eqx.filter_jit
    def loss_fn(predictor,  batch_rxns, batch_yields,  cos_freq, sin_freq, positions_padded, cache_k, cache_v, dropout_key ):

        #batch_rxns = jnp.array(batch_rxns.numpy())
        #batch_yields = jnp.array(batch_yields.numpy())

        # Pass is_training=True during training
        predictions = predictor(batch_rxns,  cos_freq, sin_freq, positions_padded, cache_k, cache_v, dropout_key, True)

        loss = jnp.mean((predictions - batch_yields) ** 2)

        return loss

    #Step 2 : Training step
    @eqx.filter_jit
    def train_step(predictor,  batch_rxns, batch_yields, cos_freq, sin_freq, positions_padded, cache_k, cache_v, dropout_key, opt_state):
        # Get loss and gradients
        loss, grads = loss_fn(predictor, batch_rxns, batch_yields,  cos_freq, sin_freq, positions_padded, cache_k, cache_v, dropout_key)
        # Update the parameters using the optimizer
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(predictor, eqx.is_array) )
        predictor = eqx.apply_updates(predictor, updates)

        return loss, predictor, opt_state

    # Step 3: Initialize optimizer
    print("\n Setting up optimizer with a learning rate = ", lr)
    optimizer = optax.adafactor(learning_rate=lr)
    opt_state = optimizer.init(eqx.filter(predictor, eqx.is_array))  # optimizer.init(predictor)
    
    # Step 4: Run Fine tuning / training
    num_epochs = args.epoch
    print("\n Total number of training epochs : ", num_epochs )
    print("\n Total number of train  batches : ",len(train_loader))
    # Training Loop with DataLoader
    for epoch in range(num_epochs):
        running_loss = 0.0
        step = 1 
        for batch_rxns, batch_yields in train_loader:
            batch_rxns = jnp.array(batch_rxns.numpy())
            batch_yields = jnp.array(batch_yields.numpy(),dtype=jnp.bfloat16)
            # Reset KV cache for each batch to ensure it's not reused across different batches
            cos_freq, sin_freq, positions_padded, cache_k, cache_v = Mistral._precompute(max_len)

            # Perform a training step
            loss, predictor, opt_state  = train_step( predictor,  batch_rxns, batch_yields, cos_freq, sin_freq, positions_padded, cache_k, cache_v, mha_dropout_key, opt_state)
        
            loss = loss.item()
            print(f"step={step}, loss={loss}")
            # Accumulate loss for monitoring
            running_loss += loss
            step += 1 
        # Print epoch loss
        print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader)}")


    # Step 5 : After training loop (i.e. validation).
    print("\nTotal number of validation batches : ",len(val_loader))
    running_val_loss = 0.0
    running_r2_score = 0.0
    all_predictions = []
    all_true_labels = []

    step = 1
    for val_rxns, val_yields in val_loader:
        val_rxns = jnp.array(val_rxns.numpy())
        val_yields = jnp.array(val_yields.numpy(),dtype=jnp.bfloat16)

        # Reset KV cache for each batch to ensure it's not reused across different batches
        cos_freq, sin_freq, positions_padded, cache_k, cache_v = Mistral._precompute(max_len)

        # Calculate embeddings and predictions without updating the model
        predictions = compute_val(predictor,  val_rxns, cos_freq, sin_freq, positions_padded, cache_k, cache_v )

        # Calculate validation loss
        val_loss = jnp.mean((predictions - val_yields) ** 2)
        running_val_loss += val_loss.item()

        # Collect predictions and true labels
        all_predictions.append(np.asarray(jnp.mean(predictions, axis=1).squeeze()))  # Flatten predictions
        all_true_labels.append(np.asarray(val_yields))

        # Calculate R² score
        #r2 = r2_score( np.asarray(val_yields), np.asarray( predictions[:, -1, 0]  ) ) # Use just the last dim of pred
        r2_alt = r2_score( np.asarray(val_yields,  dtype=np.float32), np.asarray( jnp.mean(predictions, axis=1).squeeze(), dtype=np.float32  ) )
        print("Step , MAE Loss, R2 value for current batch :", step, val_loss, r2_alt)
        running_r2_score += r2_alt
        step += 1

    # Flatten collected lists to get overall predictions and true labels
    all_predictions = np.concatenate(all_predictions, axis=0, dtype=np.float32)
    all_true_labels = np.concatenate(all_true_labels, axis=0, dtype=np.float32)

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
    print(f"Validation RMSE: {rmse}")
    print(f"Validation R² Score: {r2}")

    val_loss, r2_val=  running_val_loss / len(val_loader),  running_r2_score / len(val_loader)

    print(f"[Double Check!]Validation Loss: {val_loss}")
    print(f"[Double Check!] Validation R2: {r2_val}")


# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()
