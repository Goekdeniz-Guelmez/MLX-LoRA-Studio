# Acknowledgements

MLX LoRA Studio is built on research and open-source software from people working across efficient training, preference learning, quantization, and Apple Silicon.

The desktop app delegates fine-tuning to the [mlx-lm-lora Python package](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora). That trainer is built with [Apple's MLX array framework](https://github.com/ml-explore/mlx) and draws on the upstream [MLX LM package](https://github.com/ml-explore/mlx-lm). The runnable walkthroughs are in the [example notebook collection](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks).

## Research references

These papers describe the research foundations or closely related formulations behind the methods in this project. The implementation notes in each guide explain where Studio's trainer has additional controls or differs from a paper's original setup.

| Method | Research reference |
|---|---|
| Supervised fine-tuning and full fine-tuning | [Ouyang et al. (2022), “Training language models to follow instructions with human feedback”](https://arxiv.org/abs/2203.02155). The paper describes the supervised demonstration stage of InstructGPT; full fine-tuning is a training mode rather than a distinct algorithm. |
| Dynamic Fine-Tuning | [Wu et al. (2025), “On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification”](https://arxiv.org/abs/2508.05629). |
| LoRA | [Hu et al. (2021), “LoRA: Low-Rank Adaptation of Large Language Models”](https://arxiv.org/abs/2106.09685). |
| DoRA | [Liu et al. (2024), “DoRA: Weight-Decomposed Low-Rank Adaptation”](https://arxiv.org/abs/2402.09353). |
| QLoRA | [Dettmers et al. (2023), “QLoRA: Efficient Finetuning of Quantized LLMs”](https://arxiv.org/abs/2305.14314). |
| Quantization-aware training | [Jacob et al. (2018), “Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference”](https://arxiv.org/abs/1712.05877). The Studio guide also describes the adapter-specific fake-quantization hook used by its trainer. |
| DPO | [Rafailov et al. (2023), “Direct Preference Optimization: Your Language Model is Secretly a Reward Model”](https://arxiv.org/abs/2305.18290). |
| CPO | [Xu et al. (2024), “Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation”](https://arxiv.org/abs/2401.08417). That paper studies CPO for machine translation; see the [Studio CPO guide](CPO) for this trainer's objective and implementation. |
| FTPO | [Paech et al. (2025), “Antislop: A Comprehensive Framework for Identifying and Eliminating Repetitive Patterns in Language Models”](https://arxiv.org/abs/2510.15061) introduces Final Token Preference Optimization; [Liquid AI's Antidoom workflow](https://www.liquid.ai/blog/antidoom) describes a related implementation and data workflow. |
| ORPO | [Hong, Lee, and Thorne (2024), “ORPO: Monolithic Preference Optimization without Reference Model”](https://arxiv.org/abs/2403.07691). |
| GRPO | [Shao et al. (2024), “DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models”](https://arxiv.org/abs/2402.03300). |
| KLPO | [Zhang (2026), “On KL-Regularized Policy Optimization”](https://arxiv.org/abs/2610.08963). |
| Online DPO | The Studio loop applies the [DPO objective](https://arxiv.org/abs/2305.18290) to pairs sampled and judged during training. |
| XPO | [Xie et al. (2024), “Exploratory Preference Optimization: Harnessing Implicit Q*-Approximation for Sample-Efficient RLHF”](https://arxiv.org/abs/2405.21046). |
| REINFORCE | [Williams (1992), “Simple statistical gradient-following algorithms for connectionist reinforcement learning”](https://doi.org/10.1007/BF00992696). |
| PPO | [Schulman et al. (2017), “Proximal Policy Optimization Algorithms”](https://arxiv.org/abs/1707.06347). |
| Adam and AdamW | [Kingma and Ba (2014), “Adam: A Method for Stochastic Optimization”](https://arxiv.org/abs/1412.6980); [Loshchilov and Hutter (2017), “Decoupled Weight Decay Regularization”](https://arxiv.org/abs/1711.05101). |
| Muon | [Jordan et al. (2024), “Muon: An optimizer for hidden layers in neural networks”](https://kellerjordan.github.io/posts/muon/). |

## Project-specific methods

DSLA is a project-specific method documented in the [Studio DSLA guide](DSLA). The [Studio FTPO guide](FTPO) documents how this trainer implements Final Token Preference Optimization.

- [FTPO trainer source](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora/blob/main/mlx_lm_lora/trainer/ftpo_trainer.py)

## Software citation

The MLX project requests the following software citation:

> Awni Hannun, Jagrit Digani, Angelos Katharopoulos, and Ronan Collobert. 2023. *MLX: Efficient and flexible machine learning on Apple silicon*. [MLX project](https://github.com/ml-explore/mlx).

For implementation details and version-specific behavior, see the [training method guides](Algorithm-Guide), the [mlx-lm-lora source](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora), and the [notebook examples](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks).
