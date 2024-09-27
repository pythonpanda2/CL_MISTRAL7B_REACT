## INSTALLATION
conda create -p /path/MISTRAL_CL_CONDA  python=3.10 pip

To activate this environment, use
```
    $ conda activate /path/MISTRAL_CL_CONDA
```

To deactivate an active environment, use
```
    $ conda deactivate
```

Update and install packages. 
```
conda update -n base -c conda-forge conda

conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda install conda-forge::rdkit
```

Install cuda using conda
```
 conda install nvidia/label/cuda-11.8.0::cuda
```

Install cudnn using conda
```
conda install anaconda::cudnn

pip install --upgrade "jax[cuda11]==0.4.24"

conda install cuda-nvcc  -c nvidia
pip install --upgrade  jaxlib==0.4.24+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install equinox==0.11.4 jaxtyping==0.2.25 ml-dtypes==0.3.2 ruff==0.2.2 numpy==1.26.4 sentencepiece==0.1.99 typeguard==2.13.3 optax==0.2.2 openpyxl scikit-learn 

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
