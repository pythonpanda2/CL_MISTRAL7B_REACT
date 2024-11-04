#!/bin/bash
#PBS -l select=1:system=sophia
#PBS -l place=scatter
#PBS -l walltime=04:00:00
#PBS -q bigmem
#PBS -A FOUND4CHEM
#PBS -N TestQ
#PBS -l filesystems=home:eagle
hostname
echo $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
#module purge
#module use /soft/modulefiles
#module load conda; conda activate
#echo " "
#conda activate /eagle/FOUND4CHEM/software/conda_mace_LR

#export PATH=/eagle/FOUND4CHEM/software/conda_mace_LR/bin:$PATH
#export PYTHONPATH=/eagle/FOUND4CHEM/software/conda_mace_LR/fourier_attention_mace/mace:$PYTHONPATH
ulimit -s unlimited
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export CUDA_VISIBLE_DEVICES=0
sh ./fine_tune_launch.sh 

wait
date
