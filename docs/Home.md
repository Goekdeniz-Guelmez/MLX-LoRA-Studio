# MLX-LoRA-Studio Wiki

**Fine-tune Apple MLX models on your Mac — privately, locally, and reproducibly.**

MLX-LoRA-Studio is a macOS app for training and adapting large language models with Apple's MLX framework. No GPU farm, no cloud, no data leaving your machine. This wiki is the companion guide to the four training methods the app supports, plus the synthetic-data pipeline that feeds them.

## Where to start

New here? Read in roughly this order:

1. **[Synthetic Dataset Generation](SYNTHETIC_DATASET_GENERATION)** — make your own training data on your own machine. This is the on-ramp for everything else.
2. **[Supervised Fine-Tuning (SFT)](SFT_TRAINING)** — teach a model your domain from labeled examples. The most common starting point.
3. **[ORPO Training](ORPO_TRAINING)** — preference tuning without a reference model. Lighter than DPO, simpler than RLHF.
4. **[GRPO Training](GRPO_TRAINING)** — reinforcement learning with a reward function you can actually read and audit.
5. **[RLHF Training](RLHF_TRAINING)** — the full human-feedback loop, run locally so your most sensitive judgement never leaves the building.

## The local-first thesis

Every guide here is built on one idea: the most valuable data in your organization is the data that never leaves your building — written by a model you own, on a computer you control, for a purpose only you know. MLX-LoRA-Studio is the tool that makes that practical on Apple Silicon.

## See also

- [Main repository](https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio) — code, releases, and issues
- Use the **sidebar** (right) to jump between guides.