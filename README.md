# Continual Learning On LLM for Chemical Reactions

The Mistral-7B model weights needs to be ported to JAX/Equinox. This can be done by following the steps described in the [mistral_jax](https://github.com/AakashKumarNain/mistral_jax/blob/main/instructions.md) repo. Alternatively, one can also download them from [here](https://drive.google.com/drive/folders/1zsmyBa-7XeT8lclDwIiF2B4ZBo6HaCsN?usp=share_link). 


### Running the code
Check out the scripts folder for bash run scripts.  We can run the standard fine tuning under three different settings. The first setting is to keep the Mistral 7B weights fixed and then only allow the MHA regression head to be fine tuned. 

```
python -u  /path/CL_MISTRAL7B_REACT/CL_LLM_REACT/fine_tune_MHA_head.py -p /path/CL_MISTRAL7B_REACT/model_files \
 -xl /path/CL_MISTRAL7B_REACT/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx \
 -N 2 \
 -rs 1
```

The second setting is to jointly fine tune Mistral 7B + MHA regression head.

```
python -u  /path/CL_MISTRAL7B_REACT/CL_LLM_REACT/Jointly_fine_tune_Mistral7B_and_MHA_head.py \
 -p /path/CL_MISTRAL7B_REACT/model_files \
 -xl /path/CL_MISTRAL7B_REACT/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx \
 -N 2 \
 -rs 1

```


The third setting is to jointly fine tune Mistral 7B + MHA regression head with Low-Rank Adaptation (LoRA).

```
Work in Progress.............
```
