# Open example notebooks

The companion [mlx-lm-lora-example-notebooks repository](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks) contains editable examples for the open-source mlx-lm-lora Python package. GitHub renders each notebook in the browser; use **Download raw file** to run a local copy on Apple Silicon.

MLX LoRA Studio is a native Mac interface around the [mlx-lm-lora Python package](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora). Use the app for a guided workflow with run history, resource estimates, and live metrics, or use the notebooks to work directly with the package in code.

Browse [all of the maintainer's GitHub projects](https://github.com/Goekdeniz-Guelmez) or go directly to the [notebook repository](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks) to explore its supervised, preference, and reinforcement-learning folders.

## Supervised fine-tuning

- [Qwen3 4B instruction fine-tuning with long context](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks/blob/main/finetuning/Qwen3_4B_Instruct_32k.ipynb)
- [Qwen3.5 0.8B thinking fine-tuning](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks/blob/main/finetuning/Qwen3_5-0_8B_Thinking_32k.ipynb)
- [LFM2.5 1B instruction fine-tuning](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks/blob/main/finetuning/LFM2_5_1B_Instruct_4k.ipynb)
- [OLMo 3 with Quantization-Aware Training](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks/blob/main/finetuning/Olmo_3_4bit_QAT.ipynb)
- [Qwen3 8B fine-tuning](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks/blob/main/finetuning/Qwen3_8B_8k.ipynb)
- [Tiny Aya instruction model from scratch](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks/blob/main/finetuning/Tiny_Aya_Instruct_From_Scratch.ipynb)

## Preference optimization

- [Qwen3 4B Direct Preference Optimization](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks/blob/main/preference/Qwen3_4B_Gabliterated_DPO.ipynb)
- [Qwen3 4B Odds Ratio Preference Optimization](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks/blob/main/preference/Qwen3_4B_Gabliterated_ORPO.ipynb)
- [Gemma 3 Online DPO](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks/blob/main/preference/Gemma_3_1B-IT_Gabliterated_Online_DPO.ipynb)

## Reinforcement learning

- [Qwen3.5 0.8B with GRPO](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks/blob/main/rl/Qwen3_5-0_8B_GRPO.ipynb)
- [Tiny Aya with GRPO](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks/blob/main/rl/Tiny_Aya_GRPO.ipynb)
- [Granite 4 350M with GRPO](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks/blob/main/rl/Granite_4-0_350M_Gabliterated_GRPO.ipynb)

## Working with a notebook

1. Open a notebook above and review the model, dataset, and training settings.
2. Download the raw notebook or clone the example repository to your Mac.
3. Follow the environment instructions in the notebook and confirm the model fits your available memory.
4. Start with a short run, then inspect generations and held-out results before increasing the training budget.

Notebook defaults and model availability can change as the trainer evolves. Read the current repository README and check each notebook's setup instructions before running it.

[Choose a method first](Algorithm-Guide) · [Read the trainer documentation](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora)
