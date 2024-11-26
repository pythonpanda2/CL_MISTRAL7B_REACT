#!/bin/bash
# -----------------------------------------------------
# -----------------------------------------------------
# The following is for running on alcf nodes
export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128

#-------------------------------------------------------
## The following is for running on JLSE
#module load conda/2024.06-1
## set env
source /home/gsivaraman/miniconda3/etc/profile.d/conda.sh
conda activate py_gh200

export PATH=/home/gsivaraman/miniconda3/envs/py_gh200/bin:$PATH
export PYTHONPATH=/vast/users/gsivaraman/CL_4_RXN/CL_MISTRAL7B_REACT/CL_LLM_REACT:$PYTHONPATH

sourcepath=/vast/users/gsivaraman/CL_4_RXN/CL_MISTRAL7B_REACT/
export JAX_PLATFORMS='cuda'

nvidia-smi

start_time=`date +%s`

python -u  $sourcepath/CL_LLM_REACT/task_aware_fine_tune_Mistral7B_and_MHA_head_with_LORA_no_CL.py  -p $sourcepath/model_files  -xl $sourcepath/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx  -N 27   -rs 7193  -r 16 -s 4  -lr 2e-4 -nh 4 


end_time=`date +%s`
echo Training  time was `expr $end_time - $start_time` s.

conda deactivate 
