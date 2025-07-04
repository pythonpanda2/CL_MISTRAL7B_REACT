## INSTALLATION on ALCF SOPHIA/ POLARIS

We will leverage the [NVIDIA specific conda setup provided by ALCF](https://docs.alcf.anl.gov/polaris/data-science-workflows/frameworks/jax/). 

```
module load conda; conda activate

conda create --prefix=/path/MISTRAL_CL_CONDA  --clone base 
```


To activate this environment, use
```
    $ conda activate /path/MISTRAL_CL_CONDA
```

To deactivate an active environment, use
```
    $ conda deactivate
```

Update and install additional packages. 
```
conda install conda-forge::rdkit

pip install equinox   ruff sentencepiece    openpyxl quax 
```

## INSTALLATION on ANL JLSE : NVIDIA GH200
All the production runs were performed on NVIDIA GH200.  We used a Miniconda for 'Linux-aarch64'. Next steps are easy.  

```
conda env create -f env/env_gh200.yml 
```

To activate this environment, use
```
$ conda activate py_gh200
```
To deactivate an active environment, use
```
$ conda deactivate
```

