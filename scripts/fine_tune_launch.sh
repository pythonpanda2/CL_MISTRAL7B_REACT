#!/bin/bash 

module purge
module load conda; conda activate
conda activate /eagle/FOUND4CHEM/software/MISTRAL_CL_CONDA

export PATH=/eagle/FOUND4CHEM/software/MISTRAL_CL_CONDA/bin:$PATH
export PYTHONPATH=/eagle/FOUND4CHEM/project/CL_4_RXN/CL_MISTRAL7B_REACT/CL_LLM_REACT:$PYTHONPATH

which python
python --version
start_time=`date +%s`

python -u  /eagle/FOUND4CHEM/project/CL_4_RXN/CL_MISTRAL7B_REACT/CL_LLM_REACT/Jointly_fine_tune_Mistral7B_and_MHA_head.py  -p /eagle/FOUND4CHEM/project/CL_4_RXN/CL_MISTRAL7B_REACT/model_files  -xl /eagle/FOUND4CHEM/project/CL_4_RXN/CL_MISTRAL7B_REACT/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx  -N 2   -rs 1


end_time=`date +%s`
echo Training  time was `expr $end_time - $start_time` s.
