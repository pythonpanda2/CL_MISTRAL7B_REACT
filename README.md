# Continual Learning On LLM for Chemical Reactions
Large Language Models based on transformer architecture has shown impressive ability to learn and generate internet scale text data and beyond. In many read world example, internet scale training data might not be available. An example could be a synthetic lab chemical reaction experiments or autonomus lab experiments where continous batched streams of chemical reaction data are extracted and a model needs to be constantly updated its knowledge as the experiment knowledge evolves while minimizing forgetting of the past chemical knowledge. 
Continual learning or lifelong learning is a machine learning paradigm that allows models to learn and evolve continuously while retaining prior learned knowledge. Continual learning paradigms are particularly important when the model has to learn from an every evoling non-stationary streams of information such as our chemistry example discussed above. 

We will utilize the Mistral-7B, a 7.3B parameter open-weight large language model built using the transformers architecture. We will modify and adopt  Mistral-7B to learn chemical reaction yield of a Suzuki  Coupling Reaction. 

### Porting Model Weights to JAX
The Mistral-7B model weights needs to be ported to JAX/Equinox. This can be done by following the steps described in the [mistral_jax](https://github.com/AakashKumarNain/mistral_jax/blob/main/instructions.md) repo. Alternatively, one can also download them from [here](https://buffalo.box.com/s/ljd66kpkgte8duofz3us2zihb70btwww). 

### Installation
Check out the [INSTALL.MD](https://github.com/pythonpanda2/CL_MISTRAL7B_REACT/blob/main/INSTALL.MD) to see details of running the training scripts. 


### Running 
Check out the [RUN.md](https://github.com/pythonpanda2/CL_MISTRAL7B_REACT/blob/main/RUN.MD) to see details of running the training scripts. 
