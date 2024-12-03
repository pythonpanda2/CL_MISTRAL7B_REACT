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
import csv

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

def plot_true_vs_predicted(true_labels, predicted_labels, rmse, r2, filename="true_vs_pred_yield.png", title="True vs Predicted Yield"):
    plt.figure(figsize=(10, 6))
    plt.scatter(true_labels, predicted_labels, color='b', alpha=0.6, s=10, label='Predicted vs True')
    plt.plot([true_labels.min(), true_labels.max()], [true_labels.min(), true_labels.max()], 'k--', lw=2)
    plt.xlabel('True Yield', fontsize=12)
    plt.ylabel('Predicted Yield', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend([f'RMSE: {rmse:.2f}\nR²: {r2:.2f}'], loc='upper left')
    plt.grid(True)
    plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

def write_metric_to_csv(filename, metric_matrix, num_tasks):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        header = ['After_Task'] + [f'Task_{j+1}' for j in range(num_tasks)]
        writer.writerow(header)
        # Write data rows
        for i in range(num_tasks):
            row = [f'Task_{i+1}'] + metric_matrix[i, :].tolist()
            writer.writerow(row)

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
    max_len =     max(len(sublist) for sublist in tokenized_rxn) 
    print(f"\n Sequence length: {max_len}")
    del tokenized_rxn # We dont need this after the max_len is estimated!
    
    # 70/30 split
    train_df, test_df = train_test_split(task_aware_reactions_df, test_size=0.3, random_state=seed)
    def convert_df_to_token_y(inp_df):
        tokenize_inp_rxn = [Mistral.tokenize_smiles(smiles) for smiles in inp_df['rxn']]
        padded_inp_rxn = [sublist + [0] * (max_len - len(sublist)) for sublist in tokenize_inp_rxn]
        x_rxn_npy =  np.stack([np.array(aa) for aa in padded_inp_rxn])
        y_yields_npy = np.array(inp_df['y'], dtype=np.float32)
        return x_rxn_npy, y_yields_npy

    def create_data_loader(x_rxn_npy, y_yields_npy):
        """ Convert the np arrays to torch data loader object"""
        x_rxns_torch = torch.tensor(x_rxn_npy)
        y_yields_torch = torch.tensor(y_yields_npy, dtype=torch.float32)
        rxn_dataset = ReactionDataset(x_rxns_torch, y_yields_torch)
        rxn_loader = DataLoader(rxn_dataset, batch_size=Mistral.args.max_batch_size, shuffle=True, collate_fn=pad_collate_fn)
        return rxn_loader
    print("\n Batch size:",Mistral.args.max_batch_size)

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
        loss =  optax.losses.l2_loss(predictions.squeeze(), batch_yields) 

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

    # Initialize a list to store task data
    task_data_list = []

    # Training Loop with DataLoader
    task_losses = []  # List to store average loss per task
    task_keys = list(task_grouped_train.keys())
    # Initialize evaluation matrices
    num_tasks = len(task_grouped_train)
    metrics = {
    'avg_eval_loss': np.full((num_tasks, num_tasks), np.nan),
    'rmse': np.full((num_tasks, num_tasks), np.nan),
    'r2': np.full((num_tasks, num_tasks), np.nan)
    }

    for i, key in enumerate(task_keys ) :
        # Prepare the data for the current task
        task_df = pd.DataFrame(task_grouped_train[key])
        print(f"Task {i+1}: {key}, Samples Size: {task_df.shape[0]}")

        # Tokenization and padding
        train_rxn, train_yields = convert_df_to_token_y(task_df) # numpy arrays
        # Store the task data for later evaluation
        task_data_list.append((train_rxn, train_yields))

        #Convert to torch tensors
        train_loader = create_data_loader(train_rxn, train_yields)
        epoch_losses = []  # List to store epoch losses for this task
        for epoch in range(num_epochs):
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

        # We are sequentially overfitting each task in to the model. Hence the last epoch is the corresponding task loss!
        task_losses.append(epoch_losses[-1])
        print(f"Task {i+1} Completed. Last Epoch Loss: {np.round(epoch_losses[-1], 6)}\n")
        
        # **Save Epoch Losses for the Current Task**
        with open(f"epoch_losses_task_{i+1}.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for epoch_idx, loss_value in enumerate(epoch_losses):
                writer.writerow([epoch_idx+1, loss_value])
        print(f"Epoch losses for Task {i+1} saved to 'epoch_losses_task_{i+1}.csv'.")
        
        # Evaluate on all previous tasks (including the current one)
        for j in range(i + 1):  # i + 1 because Python indices start from 0
            eval_task_rxn, eval_task_yields = task_data_list[j]

            # Create a DataLoader for the evaluation data
            eval_loader = create_data_loader(eval_task_rxn, eval_task_yields) # DataLoader(eval_dataset,batch_size=Mistral.args.max_batch_size,shuffle=False,collate_fn=pad_collate_fn)

            # Initialize variables for evaluation
            total_eval_loss = 0.0
            total_samples = 0
            task_predictions = []
            task_true_labels = []

            for val_rxns, val_yields, mask in eval_loader:
                val_rxns = jnp.array(val_rxns.numpy())
                val_yields = jnp.array(val_yields.numpy(), dtype=jnp.float32)
                mask = jnp.array(mask.numpy(), dtype=jnp.float32)

                # Reset KV cache for each batch
                cos_freq, sin_freq, positions_padded, cache_k, cache_v = Mistral._precompute(max_len)

                # Calculate predictions without updating the model
                predictions = compute_val(params,static,val_rxns,cos_freq,sin_freq,positions_padded,cache_k,cache_v)

                # Compute per-sample losses
                losses =  optax.losses.l2_loss(predictions.squeeze(), val_yields) 

                losses = loss * mask  # Zero out losses for padded samples

                # Sum the losses and count real samples
                batch_loss = jnp.sum(losses)
                batch_real_samples = jnp.sum(mask)
        
                # Accumulate total loss and total samples
                total_eval_loss += batch_loss.item()
                total_samples += batch_real_samples.item()
                
                # Collect predictions and true labels for real samples
                real_predictions = np.asarray(predictions.squeeze(), dtype=np.float32)[mask.astype(bool)]
                real_true_labels = np.asarray(val_yields, dtype=np.float32)[mask.astype(bool)]

                task_predictions.append(real_predictions)
                task_true_labels.append(real_true_labels)

            # Calculate average loss for the current evaluation task
            avg_eval_loss = total_eval_loss / total_samples
            metrics['avg_eval_loss'][i, j] = avg_eval_loss  # Use the avg_eval_loss computed in the loop above

            # Optionally, you can compute RMSE and R² score here
            task_predictions = np.concatenate(task_predictions, axis=0)
            task_true_labels = np.concatenate(task_true_labels, axis=0)
            rmse = mean_squared_error(task_true_labels, task_predictions, squared=False)
            r2 = r2_score(task_true_labels, task_predictions)
            metrics['rmse'][i, j] = rmse
            metrics['r2'][i, j] = r2
            print(f"After training Task {i+1}, Evaluation on Task {j+1}: Loss={np.round(avg_eval_loss,6)}, RMSE={np.round(rmse,5)}, R²={np.round(r2,5)}")
    print(f"Evaluation matrix updated and saved after Task {i+1}.")
    for metric_name, metric_matrix in metrics.items():
        filename = f'{metric_name}_train_task_eval_matrix.csv'
        write_metric_to_csv(filename, metric_matrix, num_tasks)
        print(f"{metric_name.capitalize()} matrix saved to '{filename}'.")


    # **Save All Task Losses**
    with open("task_losses.csv", 'w', newline='') as taskcsv:
        writer = csv.writer(taskcsv)
        for task_idx, loss_value in enumerate(task_losses):
            writer.writerow([task_idx+1, loss_value])
    print("All task losses saved to 'task_losses.csv'.")

    # Step 5 : After training loop (i.e. validation).
    # === Evaluation on the Test Set ===
    print("\nEvaluating on the 30% test set.")

    # Split the test_df into task-specific dataframes
    task_grouped_test = task_aware_splits(test_df)
    num_test_tasks = len(task_grouped_test)
    print(f"\nTotal number of test task groups: {num_test_tasks}")

    test_metrics = {
        'avg_eval_loss': np.full((num_test_tasks + 1,), np.nan),
        'rmse': np.full((num_test_tasks + 1,), np.nan),
        'r2': np.full((num_test_tasks + 1,), np.nan) }

    overall_predictions = []
    overall_true_labels = []

    # Evaluate on each test task
    for i, key in enumerate(task_grouped_test.keys()):
        task_df = pd.DataFrame(task_grouped_test[key])
        print(f"\nEvaluating on Test Task {i+1}: {key}, Sample Size: {task_df.shape[0]}")

        # Tokenize and pad the test data
        test_rxn, test_yields = convert_df_to_token_y(task_df)
        test_loader = create_data_loader(test_rxn, test_yields)

        # Initialize variables for evaluation
        total_eval_loss = 0.0
        total_samples = 0
        task_predictions = []
        task_true_labels = []

        for val_rxns, val_yields, mask in test_loader:
            val_rxns = jnp.array(val_rxns.numpy())
            val_yields = jnp.array(val_yields.numpy(), dtype=jnp.float32)
            mask = jnp.array(mask.numpy(), dtype=jnp.float32)

            # Reset KV cache for each batch
            cos_freq, sin_freq, positions_padded, cache_k, cache_v = Mistral._precompute(max_len)

            # Calculate predictions
            predictions = compute_val(params, static, val_rxns, cos_freq, sin_freq, positions_padded, cache_k, cache_v)

            # Compute per-sample losses
            losses = optax.losses.l2_loss(predictions.squeeze(), val_yields)
            losses = losses * mask  # Zero out losses for padded samples

            # Sum the losses and count real samples
            batch_loss = jnp.sum(losses)
            batch_real_samples = jnp.sum(mask)

            # Accumulate total loss and total samples
            total_eval_loss += batch_loss.item()
            total_samples += batch_real_samples.item()

            # Collect predictions and true labels for real samples
            real_predictions = np.asarray(predictions.squeeze(), dtype=np.float32)[mask.astype(bool)]
            real_true_labels = np.asarray(val_yields, dtype=np.float32)[mask.astype(bool)]

            task_predictions.append(real_predictions)
            task_true_labels.append(real_true_labels)

        # Calculate average loss for the current test task
        avg_eval_loss = total_eval_loss / total_samples
        test_metrics['avg_eval_loss'][i] = avg_eval_loss

        # Concatenate predictions and true labels
        task_predictions = np.concatenate(task_predictions, axis=0)
        task_true_labels = np.concatenate(task_true_labels, axis=0)

        # Collect for overall metrics
        overall_predictions.append(task_predictions)
        overall_true_labels.append(task_true_labels)

        # Compute RMSE and R² score for the current task
        rmse = mean_squared_error(task_true_labels, task_predictions, squared=False)
        r2 = r2_score(task_true_labels, task_predictions)
        test_metrics['rmse'][i] = rmse
        test_metrics['r2'][i] = r2

        print(f"Test Task {i+1}, Evaluation: Loss={np.round(avg_eval_loss,6)}, RMSE={np.round(rmse,5)}, R²={np.round(r2,5)}")

        # Save predictions and true labels for the task
        np.save(f"test_true_labels_task_{i+1}.npy", task_true_labels)
        np.save(f"test_predictions_task_{i+1}.npy", task_predictions)

        # Plot true vs predicted for this task
        plot_filename = f"true_vs_predicted_test_task_{i+1}.png"
        plot_title = f"True vs Predicted Yield for Task {i+1}"
        plot_true_vs_predicted(task_true_labels, task_predictions, rmse, r2, plot_filename, plot_title)

    # Concatenate overall predictions and true labels
    overall_predictions = np.concatenate(overall_predictions, axis=0)
    overall_true_labels = np.concatenate(overall_true_labels, axis=0)

    # Compute overall metrics
    overall_loss = mean_squared_error(overall_true_labels, overall_predictions)
    overall_rmse = mean_squared_error(overall_true_labels, overall_predictions, squared=False)
    overall_r2 = r2_score(overall_true_labels, overall_predictions)

    # Store the overall metrics
    test_metrics['avg_eval_loss'][-1] = overall_loss
    test_metrics['rmse'][-1] = overall_rmse
    test_metrics['r2'][-1] = overall_r2

    print(f"\nOverall Test Evaluation: Loss={np.round(overall_loss,6)}, RMSE={np.round(overall_rmse,5)}, R²={np.round(overall_r2,5)}")

    # Save the overall true labels and predictions
    np.save("test_true_labels.npy", overall_true_labels)
    np.save("test_predictions.npy", overall_predictions)

    # Plot overall true vs predicted
    plot_true_vs_predicted(overall_true_labels, overall_predictions, overall_rmse, overall_r2, "true_vs_predicted_test_overall.png")
    # After computing overall metrics
    plot_true_vs_predicted(overall_true_labels, overall_predictions, overall_rmse, overall_r2, filename="true_vs_predicted_test_overall.png", title="True vs Predicted Yield for Overall Test Set" )
    # Prepare task names
    task_names = [f'Test_Task_{i+1}' for i in range(num_test_tasks)] + ['Overall']

    # Create a DataFrame for the test metrics
    test_metrics_df = pd.DataFrame({
        'Task': task_names,
        'Loss': test_metrics['avg_eval_loss'],
        'RMSE': test_metrics['rmse'],
        'R2': test_metrics['r2']
        })

    # Save the test metrics to a CSV file
    test_metrics_df.to_csv("test_metrics.csv", index=False)
    print("Test metrics saved to 'test_metrics.csv'.")

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()
