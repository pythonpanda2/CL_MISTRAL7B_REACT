# Continual Learning On LLM for Chemical Reactions

The Mistral-7B model weights needs to be ported to JAX/Equinox. This can be done by following the steps described in the [mistral_jax](https://github.com/AakashKumarNain/mistral_jax/blob/main/instructions.md) repo. Alternatively, one can also download them from [here](https://buffalo.box.com/s/ljd66kpkgte8duofz3us2zihb70btwww). 


### Running the code
Check out the [scripts](https://github.com/pythonpanda2/CL_MISTRAL7B_REACT/tree/main/scripts) folder for bash run scripts.  We can run the standard fine tuning under three different settings. The first setting is to keep the Mistral 7B weights fixed and then only allow the MHA regression head to be fine tuned. 

```
python -u  /path/CL_MISTRAL7B_REACT/CL_LLM_REACT/fine_tune_MHA_head.py -p /path/CL_MISTRAL7B_REACT/model_files \
 -xl /path/CL_MISTRAL7B_REACT/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx \
 -N 2 \
 -rs 1
```

The second setting is to jointly fine tune Mistral 7B + MHA regression head. **Krishnan : This is the standard running with out LoRA. The `submit.sh` takes care of this. I have increased the batch size to 128  inside the `MISTRAL7B_MHA_LOADER.py`.**

```
python -u  /path/CL_MISTRAL7B_REACT/CL_LLM_REACT/Jointly_fine_tune_Mistral7B_and_MHA_head.py -p /path/CL_MISTRAL7B_REACT/model_files \
 -xl /path/CL_MISTRAL7B_REACT/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx \
 -N 2 \
 -rs 1

```


The third setting is to jointly fine tune Mistral 7B + MHA regression head with Low-Rank Adaptation (LoRA). 

```
python -u  /path/CL_MISTRAL7B_REACT/CL_LLM_REACT/Jointly_fine_tune_Mistral7B_and_MHA_head_with_LORA.py  -p /path/CL_MISTRAL7B_REACT/model_files \
 -xl /path/CL_MISTRAL7B_REACT/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx \
 -N 27  \ # Number of epochs
 -rs 7193 \ # Seed
-r 16 \ # LoRA rank
-s 4 \ # LoRA scale : The optimal  value for our dataset is  4 x rank. Provides comparable results to full fine tuning!
-lr 2e-4  #Learning rate
-nh 4 #Number of attention heads for read out regression MHA : Optimized
```

### Task-Aware Fine Tuning with LoRA with no CL

```
python -u  /path/CL_MISTRAL7B_REACT/CL_LLM_REACT/task_aware_fine_tune_Mistral7B_and_MHA_head_with_LORA_no_CL.py  -p /path/CL_MISTRAL7B_REACT/model_files \
 -xl /path/CL_MISTRAL7B_REACT/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx \
 -N 27  \ # Number of epochs
 -rs 7193 \ # Random seed
-r 16 \ # LoRA rank : optimized
-s 4 \ # LoRA scale : optimized
-lr 2e-4 \ #Learning rate
-nh 4  #Number of attention heads for read out regression MHA : Optimized
```

### Task-Aware Fine Tuning with LoRA with Experience Replay (i.e. CL)
We have different variations of the task aware expereince replay implemented in our workflow. The variations come in the form of how the gradients are computed. We do get similar results between the two implementations. 

```
python -u  /path/CL_MISTRAL7B_REACT/CL_LLM_REACT/task_aware_fine_tune_Mistral7B_and_MHA_head_with_LORA_combined_loss_grad_Experience_Replay.py  -p /path/CL_MISTRAL7B_REACT/model_files  -xl /path/CL_MISTRAL7B_REACT/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx  -N 15   -rs 7193  -r 16 -s 4  -lr 2e-4 -nh 4 
```

```
python -u  /path/CL_MISTRAL7B_REACT/CL_LLM_REACT/task_aware_fine_tune_Mistral7B_and_MHA_head_with_LORA_separate_grad_tree_Experience_Replay.py  -p /path/CL_MISTRAL7B_REACT/model_files  -xl  /path/CL_MISTRAL7B_REACT/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx  -N 15   -rs 7193  -r 16 -s 4  -lr 2e-4 -nh 4 
```


