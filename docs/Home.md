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

## Guides

The wiki is being filled in. These pages mirror the app's sidebar sections:

1. **[Train](Train)** — model & data, adapter shape, optimization, algorithm-specific fields, resume/output.
2. **[Live Metrics](Live-Metrics)** — loss/reward/KL plots, throughput, and the streaming console.
3. **[Synthetic Data](Synthetic-Data)** — prompt, SFT, and DPO generation modes.
4. **[Upload to HF](Upload-to-HF)** — repository settings, model card, token handling, push progress.
5. **[Algorithm Guide](Algorithm-Guide)** — when to use each of the 9 algorithms, key hyperparameters, failure modes.
6. **[Runs](Runs)** — the runs archive: status, config, resume, upload, reveal, delete.
7. **[Settings & Onboarding](Settings-and-Onboarding)** — Python environment, HF cache paths, the first-launch tour.

## See also

- [Main repository](https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio) — code, releases, and issues
- [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) — the underlying trainer
- Use the **sidebar** (right) to jump between guides.