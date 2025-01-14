# Continual Learning for Large Language Models in Predicting Chemical Reaction Yields
Large Language Models (LLMs) based on transformer architectures have demonstrated impressive capabilities in learning and generating text from vast datasets. However, in many real-world scenarios, such extensive training data may not be available. For instance, in synthetic chemistry labs or autonomous experimental setups, data is often generated in continuous batches as new chemical reactions are conducted. In such cases, a model needs to continuously update its knowledge as experimental insights evolve, all while minimizing the forgetting of previously learned information.

Continual learning, also known as lifelong learning, is a machine learning paradigm designed to address this challenge. It enables models to learn and adapt continuously while retaining prior knowledge. This paradigm is especially critical for tasks involving ever-evolving, non-stationary data streams, such as the chemical reaction experiments described above.

In this work, we utilize Mistral-7B (v0.1), a 7.3-billion parameter open-weight large language model constructed with 32 transformer layers. We adapt and modify Mistral-7B to predict the chemical reaction yield of Suzuki Coupling Reactions. Specifically, we enhance the original pretrained Mistral-7B by integrating a custom multi-head attention regression (MHAR) head, designed to learn chemical reaction yields. The custom MHAR head is jointly fine-tuned using both full fine-tuning and Low-Rank Adaptation (LoRA) methods to evaluate the predictive performance of this framework in a standard, epochal, end-to-end supervised training scenario where all training data is available.

In our framework, we define a task-aware learning paradigm, where each subset of data corresponds to a specific pair of reactants, forming an independent task. For example, similar to human activities such as walking, sleeping, or eating being categorized as separate tasks, a new pair of reactants defines a new task in our model. Unlike the full supervised fine-tuning approach, in the task-aware setting, the model learns each task sequentially before progressing to the next.

We leverage the optimal LoRA configuration that achieves comparable performance to full fine-tuning for task-aware fine-tuning. This setting allows us to explore the phenomenon of catastrophic forgetting, where a model loses previously acquired knowledge as it learns new tasks. By employing task-aware LoRA, we highlight the challenges of catastrophic forgetting in LLMs when processing non-stationary data streams. Furthermore, we demonstrate how experience replay can effectively mitigate catastrophic forgetting, ensuring the retention of prior knowledge while learning new tasks.

All experiments and implementations are carried out in JAX and Equinox.

### Porting Model Weights to JAX
The Mistral-7B model weights needs to be ported to JAX/Equinox. This can be done by following the steps described in the [mistral_jax](https://github.com/AakashKumarNain/mistral_jax/blob/main/instructions.md) repo. Alternatively, one can also download them from [here](https://buffalo.box.com/s/ljd66kpkgte8duofz3us2zihb70btwww). 

### Installation
Check out the [INSTALL.MD](https://github.com/pythonpanda2/CL_MISTRAL7B_REACT/blob/main/INSTALL.MD) to see details of running the training scripts. 


### Running 
Check out the [RUN.md](https://github.com/pythonpanda2/CL_MISTRAL7B_REACT/blob/main/RUN.MD) to see details of running the training scripts. 
