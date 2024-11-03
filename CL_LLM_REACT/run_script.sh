#!/bin/bash
# -----------------------------------------------------


# -----------------------------------------------------
# The following is for running on alcf nodes
export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128


#-------------------------------------------------------
## The following is for running on JLSE
module load module load conda/2024.06-1
conda activate py_gh200
python Mistral_7B_Fine_Tune.py
conda deactivate 