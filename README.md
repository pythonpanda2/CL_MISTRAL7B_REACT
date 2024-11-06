# Continual Learning On LLM for Chemical Reactions

The Mistral-7B model weights needs to be ported to JAX/Equinox. This can be done by following the steps described in the [mistral_jax](https://github.com/AakashKumarNain/mistral_jax/blob/main/instructions.md) repo. Alternatively, one can also download them from [here](https://uofi.box.com/s/ljd66kpkgte8duofz3us2zihb70btwww). 


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


The third setting is to jointly fine tune Mistral 7B + MHA regression head with Low-Rank Adaptation (LoRA). **Krishnan : This is what you should be running. Modify the  `fine_tune_launch.sh` with the following line and `submit.sh` launches the fine tune launch script. I have increased the batch size to 128  inside the `MISTRAL7B_MHA_LOADER.py`. Modify any of the paramter below as needed.**

```
python -u  /path/CL_MISTRAL7B_REACT/CL_LLM_REACT/Jointly_fine_tune_Mistral7B_and_MHA_head_with_LORA.py  -p /path/CL_MISTRAL7B_REACT/model_files \
 -xl /path/CL_MISTRAL7B_REACT/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx \
 -N 2  \ # Number of epochs
 -rs 1 \ # Seed
-r 8 \ # LoRA rank
-s 0.1 \ # LoRA scale : Typical value is 2 x rank. Kept low for now. 
-lr 1e-5  #Learning rate
```
