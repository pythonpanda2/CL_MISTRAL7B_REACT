# Continual Learning for Large Language Models in Predicting Chemical Reaction Yields

Large Language Models (LLMs) based on transformer architectures have shown remarkable capabilities in learning and generating text from vast datasets. However, in real-world scenarios like synthetic chemistry labs or autonomous experimental setups, data often arrives incrementally as new chemical reactions are conducted. In such cases, models must continuously update their knowledge while minimizing the loss of previously learned information.

Continual learning, also known as lifelong learning, addresses this challenge by enabling models to learn and adapt incrementally while retaining prior knowledge. This paradigm is especially critical for tasks involving non-stationary data streams, such as evolving chemical reaction experiments.

![](https://github.com/pythonpanda2/CL_MISTRAL7B_REACT/blob/main/figure/plot_CL.png)

## Objective

This project adapts the **Mistral-7B** (v0.1), a 7.3-billion-parameter open-weight LLM with 32 transformer layers, for predicting chemical reaction yields. Specifically, we integrate a custom **Multi-Head Attention Regression (MHAR)** head into the pretrained Mistral-7B model to enhance its capabilities in predicting chemical reaction yields.

### Key Features
- **Baseline Fine-Tuning**: The model is fine-tuned using both full fine-tuning and Low-Rank Adaptation (LoRA) methods on the Suzuki Coupling Reactions dataset. These methods serve as benchmarks for predictive performance in traditional end-to-end supervised training.
  
- **Task-Incremental Learning**: To mimic real-world scenarios where training data arrives in continuous batches, we implement a **task-aware learning paradigm**:
  - Each data subset corresponds to a distinct pair of reactants, forming a specific task group.
  - New task groups are introduced sequentially, requiring the model to learn each task while preserving knowledge from previous ones.
  - The Suzuki Coupling dataset is partitioned into these task groups to simulate incremental learning scenarios.

## Approach

1. **Baseline Training**:
   - The MHAR head is jointly fine-tuned with the pretrained Mistral-7B using both full fine-tuning and LoRA.
   - These experiments establish baseline predictive performance in standard training settings where all data is available at once.

2. **Task-Aware Learning**:
   - The optimal LoRA configuration (achieving comparable performance to full fine-tuning) is used for task-aware fine-tuning.
   - Initially, task-aware fine-tuning is performed **without any continual learning**. This setting allows us to explore the phenomenon of **catastrophic forgetting**, where a model loses previously acquired knowledge as it learns new tasks.
   - This approach highlight the challenges of **catastrophic forgetting** in LLMs when processing non-stationary data streams, where the model loses prior knowledge as it learns new tasks.

3. **Mitigating Forgetting**:
   - Finally, we demonstrate how **experience replay**, a continual learning technique can effectively mitigate catastrophic forgetting.
   - Experience replay ensures the retention of prior knowledge while learning new tasks.


## Results

This framework highlights the challenges and solutions for training LLMs on non-stationary data streams:
- **Performance Benchmarks**: Full fine-tuning and LoRA are compared in standard training settings.
- **Forgetting Analysis**: Task-aware fine-tuning explores the impact of catastrophic forgetting on LLMs.
- **Replay Effectiveness**: Experience replay is shown to be a useful continual learning technique for preserving prior knowledge during task-incremental learning.

## Implementation Details

- All experiments are implemented in **JAX** and **Equinox**, leveraging their flexibility and efficiency for neural network training.
- The codebase is modular and extensible, enabling further experimentation with continual learning techniques.


### Porting Model Weights to JAX
The Mistral-7B model weights needs to be ported to JAX/Equinox. This can be done by following the steps described in the [mistral_jax](https://github.com/AakashKumarNain/mistral_jax/blob/main/instructions.md) repo. Alternatively, one can also download them from [here](https://buffalo.box.com/s/ljd66kpkgte8duofz3us2zihb70btwww). 

### Installation
Check out the [INSTALL.MD](https://github.com/pythonpanda2/CL_MISTRAL7B_REACT/blob/main/INSTALL.MD) to see details of running the training scripts. 


### Running 
Check out the [RUN.md](https://github.com/pythonpanda2/CL_MISTRAL7B_REACT/blob/main/RUN.MD) to see details of running the training scripts. 
