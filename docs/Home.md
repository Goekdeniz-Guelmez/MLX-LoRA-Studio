# Welcome to MLX LoRA Studio

**A native Mac app for LLM fine-tuning on Apple Silicon — fully on-device, fully open source.**

MLX LoRA Studio turns fine-tuning into a normal Mac workflow: pick a model, choose a dataset, select an algorithm, watch live training metrics, generate synthetic data, and publish adapters to Hugging Face — without leaving the window, and without your data ever leaving your Mac.

It is a graphical front-end to the [`mlx-lm-lora`](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) Python training pipeline, vendored at `vendor/mlx-lm-lora/`, so what runs in the GUI is exactly what you can run from the CLI.

## Requirements

- **macOS 14 (Sonoma) or later**
- **Apple Silicon** (M1 / M2 / M3 / M4). Intel is not supported.
- **16 GB RAM minimum**; 24 GB+ recommended for ≥13B models
- ~5 GB disk for the app plus a per-model Hugging Face cache

## Features at a glance

### 🧠 Training
- **9 training algorithms:** SFT, DPO, CPO, ORPO, GRPO, Online DPO, XPO, RLHF Reinforce, and PPO.
- **5 adapter / training modes:** LoRA, DoRA, QLoRA (4/6/8-bit), full fine-tuning, and **Quantization-Aware Training (QAT)**.
- **10 optimizers selectable** (Adam, AdamW, Muon documented; plus SGD, RMSprop, Adagrad, AdaDelta, Adamax, Lion, Adafactor).
- **Adapter resume** — continue from an existing checkpoint.
- **Judge / reward model selection** for RL-style algorithms.
- **YAML-driven configuration** — the GUI form is a view over a YAML config that can also be run on the CLI.

### 📊 Live observability
- Live loss, learning rate, gradient norm, throughput, and a refreshable step plot.
- Live wired/active memory monitor plus a per-configuration memory estimate.
- Run progress bar, and **pause / resume / stop** from the toolbar.

### 🧪 Synthetic data
- Prompt generation, SFT pair generation, and DPO preference-triple generation — all with local models.
- In-app preview and JSONL export straight into the Train tab.

### 🚀 Publish
- One-click **Hugging Face upload** of adapters with model-card metadata and license pickers.
- **Runs archive** with configs, logs, adapter weights, resume, Finder reveal, and upload handoff.

### 🛠 Engineering safeguards
- **Python environment discovery & provisioning** — Studio finds or creates a working env for you.
- **ResourceGuard** — watches OS memory pressure and refuses to start a job the system can't fit, with a clear reason.
- **Self-contained app bundle** — bundled Python + trainer, so a drag-installed copy works without the source tree.

## Algorithms (detailed reference)

Grouped into three families — supervised, preference, and reinforcement/online. Each page covers the loss, the math, the dataset shape, when to use it, the settings table, and failure modes.

**Supervised**
- [SFT](SFT) — next-token cross-entropy; the substrate every other algorithm reuses.

**Preference**
- [DPO](DPO) — closed-form preference loss with a frozen reference model.
- [CPO](CPO) — DPO without the reference; lighter, more sensitive.
- [ORPO](ORPO) — SFT + odds-ratio preference in one loss, no reference, no warm-up.

**Reinforcement / online**
- [GRPO](GRPO) — group-relative advantage with user-supplied reward functions.
- [Online DPO](Online-DPO) — DPO on policy-sampled, judge-labelled pairs.
- [XPO](XPO) — Online DPO plus a KL exploration bonus.
- [RLHF-REINFORCE](RLHF-Reinforce) — classic policy-gradient RLHF, simplest, highest variance.
- [PPO](PPO) — clipped surrogate objective, most expressive, most finicky.

## Adaptation methods & foundations

Orthogonal to the loss: which tensors are trainable, how gradients become updates, and how the forward pass is quantised.

**Adaptation methods**
- [LoRA](LoRA) — low-rank adapters; the default.
- [DoRA](DoRA) — weight-decomposed LoRA (magnitude + direction).
- [Full fine-tuning](Full-Fine-Tuning) — every weight trainable.

**Quantization**
- [QLoRA](QLoRA) — load-time 4 / 6 / 8 / MXFP4-bit quantization of the base model.
- [QAT](QAT) — Quantization-Aware Training; train the adapter as if it will be deployed quantised.

**Optimizers**
- [Optimizers](Optimizers) — Adam, AdamW, Muon.

## App sections

Pages mirroring the app's sidebar:

- [Train](Train) · [Live Metrics](Live-Metrics) · [Synthetic Data](Synthetic-Data) · [Upload to HF](Upload-to-HF) · [Algorithm Guide](Algorithm-Guide) · [Runs](Runs) · [Settings & Onboarding](Settings-and-Onboarding)

## See also

- [Main repository](https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio) — code, releases, and issues
- [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) — the underlying trainer
- Use the **sidebar** (right) to jump between pages.