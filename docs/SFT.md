# Supervised Fine-Tuning (SFT)

**Family:** Supervised · **Reference model:** no · **Judge:** no · **QAT:** yes

> The canonical "teach the model to imitate a dataset" loop, and the substrate every other algorithm in MLX LoRA Studio is built on.

## Overview

**Supervised fine-tuning (SFT)** trains the model to maximise the log-probability it assigns to a completion, given a prompt. Every example is a `(prompt, completion)` pair (or a chat-style `messages` list, or a raw `text` blob). Prompt tokens are masked out so the loss only counts the answer.

In `mlx-lm-lora` the SFT trainer is the substrate that every other algorithm reuses. The cross-entropy / NLL loss, gradient checkpointing, KV-cache handling, QAT hooks, and the sequence-chunked efficient-long-context forward pass all live in `sft_trainer.py`. Every preference and RL algorithm inherits those mechanics and only swaps the loss function and (sometimes) the dataset format.

## Intuition

- The model gets a question, you give it the right answer, and you ask it to be more confident in the right answer next time.
- Masking the prompt means the loss only punishes (or rewards) the tokens the model *generated* in its answer, not the question.
- Iterations are stochastic gradient steps over mini-batches; the validation set is a held-out slice that you should never train on.
- Because everything else in this guide is built on top of SFT, the SFT trainer is also where `grad_checkpoint`, `efficient_long_context`, `seq_step_size`, and `qat_*` live.

## Objective (math)

Per-example next-token cross-entropy, masked so only the completion contributes. Implemented in `sft_trainer.default_loss`: it shifts the input by one to form `(inputs, targets)`, optionally restricts the loss to a `[length_start, length_end]` range via the cache offset (used for `efficient_long_context`), and averages over the masked token count.

```text
ℒ_sft(θ) = − ∑_t  log p_θ(y_t | prompt, y_<t)        for t ∈ completion tokens
```

## Dataset format

SFT accepts three shapes, in priority order:

1. **Chat messages** — a `messages` list of `{"role", "content"}` dicts (with an optional `system` field). The trainer applies the tokenizer's chat template.
2. **Prompt / completion** — explicit `(prompt, completion)` strings; the prompt is masked out of the loss when `mask_prompt = true`.
3. **Text** — a single `text` string for raw next-token pretraining-style fine-tuning.

The bundled default for SFT is `mlx-community/JOSIE-v2-Instruct-5K`, a 5K-row instruction-tuning set in the messages format.

Minimal SFT YAML (the same file the GUI writes, runnable on the CLI with `python -m mlx_lm_lora -c run.yaml`):

```yaml
train_mode: sft
model: mlx-community/Meta-Llama-3-8B-Instruct-4bit
dataset:
  - mlx-community/JOSIE-v2-Instruct-5K
adapter:
  type: lora
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
optim:
  learning_rate: 2.0e-4
  schedule: cosine
  warmup_steps: 20
  weight_decay: 0.0
  grad_accumulation_steps: 8
train:
  max_steps: 1000
  batch_size: 2
  max_seq_len: 2048
  save_every: 200
output:
  adapter_path: ~/Library/Application Support/MLXLoRAStudio/runs/<id>/adapter
```

## When to use it

**Almost always first.** SFT establishes the model's behaviour — tone, format, refusal style, domain knowledge. Every other algorithm assumes you have an SFT checkpoint that is already in the right ballpark for the task, and only fine-tunes alignment on top.

Reasonable defaults: `lr=1e-5`, `batch_size=1`, `grad_accum=8`, `iters=1000`, `rank=8`, `scale=20.0`.

## Shared configuration reference

These settings are the SFT substrate — every other algorithm inherits them. Each algorithm page lists only its *additional* settings on top of this table.

| Setting | Default | What it actually changes |
|---|---|---|
| `model` | `Qwen/Qwen3-0.6B` | HF id or local path of the base model. Loaded with the chosen quantisation and wrapped with LoRA/DoRA/full adapters. |
| `data` | `auto` | HF dataset id, local folder, or JSONL. The trainer's `create_dataset` picks column names per `train_mode`. |
| `train_type` | `lora` | `lora`, `dora`, or `full`. See [LoRA](LoRA), [DoRA](DoRA), [Full fine-tuning](Full-Fine-Tuning). |
| `lora_parameters.rank` | `8` | Inner rank of the LoRA adapters. Higher rank = more capacity, more parameters. 4–16 typical on a laptop. |
| `lora_parameters.scale` | `20.0` | LoRA scaling factor (α). Effective update is `(α/r) · B·A`, so the ratio of scale to rank is what matters. |
| `lora_parameters.dropout` | `0.0` | Dropout inside the LoRA branch. Only > 0 for tiny datasets or visible overfitting. |
| `num_layers` | `16` | How many transformer layers (from the top) get LoRA adapters. |
| `batch_size` | `1` | Per-device minibatch. Real batch on disk is `batch_size · gradient_accumulation_steps`. |
| `gradient_accumulation_steps` | `1` | Micro-batches to accumulate before stepping the optimiser. |
| `iters` / `epochs` | `iters=1000` | Fixed iterations or full passes. `epochs > 0` switches to epoch-counted training. |
| `learning_rate` | `1e-5` | Peak LR. 1e-5 to 5e-5 for LoRA; 1e-6 to 5e-6 for full fine-tuning. |
| `steps_per_report` | `10` | Print loss/reward every N optimiser steps. Drives the live-metrics tick. |
| `steps_per_eval` | `200` | Validation pass every N steps. `-1` disables. |
| `val_batches` | `25` | Validation minibatches per eval. `-1` uses the entire validation set. |
| `save_every` | `100` | Checkpoint adapter weights every N steps under `<adapter_path>/<iter>_adapters.safetensors`. |
| `max_seq_length` | `2048` | Hard cap on token length per example. Drives the KV-cache memory budget. |
| `grad_checkpoint` | `true` | Rewrites every layer's forward to use `mx.checkpoint`. ~halves activation memory for one extra recompute per layer. |
| `efficient_long_context` | `false` | Splits long sequences into `seq_step_size` chunks and reuses the KV-cache. Enable when `max_seq_length` is large and you OOM. |
| `seq_step_size` | `512` | Chunk size for `efficient_long_context`. 256–1024; smaller = less memory, more recompute. |
| `mask_prompt` | `false` | If true, loss is computed only over the completion. SFT and DPO enforce this internally; the flag is for custom datasets. |
| `fuse` | `true` | Merge LoRA weights back into the base model after training and save a standalone checkpoint. |
| `lm_studio_name` | `—` | If set, write the merged model into the LM Studio models directory under this name. |
| `resume_adapter_file` | `—` | Path to a saved `adapters.safetensors` to warm-start from. Useful for SFT-then-DPO pipelines. |
| `seed` | `0` | Seed for `numpy` and `mlx.core`. MLX GPU kernels are not bit-exact across driver versions. |
| `load_in_4bits` / `6` / `8` / `mxfp4` | `4-bit` | Load-time quantisation. `4-bit`/`MXFP4` most aggressive; 8-bit safe default for preference training. See [QLoRA](QLoRA). |
| `qat_enable` | `false` | Install a symmetric fake-quantise hook on every `nn.Linear` after the first optimiser step. See [QAT](QAT). |
| `qat_bits` / `group_size` / `start_step` / `interval` | `8 / 64 / 1 / 1` | Bit-width, group size (0 = per-tensor), first step to enable, re-project interval. |
| `test` / `test_batches` | `false / 100` | If `test` is true, evaluate a held-out test split after training and write the result into the run record. |

### SFT-specific settings

| Setting | Default | What it actually changes |
|---|---|---|
| `prompt_feature` / `completion_feature` | `auto` | Column-name overrides for the `(prompt, completion)` schema. |
| `messages_feature` / `system_feature` | `auto` | Column-name overrides for the chat-template `messages` schema (multi-turn). |
| `text_feature` | `auto` | Column-name override for the raw-pretraining `text` schema. |

## In the app

On the **Train** tab, SFT uses only the shared form — there is no algorithm-specific block:

- **Model & Data** — base model + dataset (HF asset pickers), LM Studio export name.
- **Fine-tune** — `train_type` (LORA / DORA / FULL); when LoRA/DoRA, **LoRA Settings** (Layers → `num_layers`, Rank, Scale, Dropout); **Quantization** (None / 4-bit / 6-bit / 8-bit / MXFP4).
- **Training Settings** — Iterations, Epochs, Batch, Max Seq, Seed, Learning Rate, Optimizer, LR Schedule (Warmup, Decay Fraction, Final LR), Report / Eval / Save, Val Batches, Gradient accumulation steps, Sequence step size (`efficient_long_context`), Test Batches; toggles for grad checkpoint, mask prompt, fuse.
- **Dataset Columns** — Prompt, Completion, Chat, Text, System (the column-name overrides for SFT's three schemas).
- **Output** — run folder name.
- **QAT** — Bits, Group, Start, Interval (toggle to enable; SFT supports QAT).

## Tips & gotchas

- Start with the bundled default dataset and run for a small number of iters — you should see `loss` drop within 100 steps. If it does not, the dataset probably has the wrong column names for SFT (the trainer raises `Unsupported data format`).
- If you are OOM-ing, enable `grad_checkpoint` first, then drop `max_seq_length`, then try `efficient_long_context` with a `seq_step_size` of 256.
- QAT is most useful when you plan to ship a 4-bit or 8-bit version of the adapter for LM Studio. Leave it off during development — it adds noticeable per-step overhead.
- `mask_prompt=true` is the default behaviour for SFT in the trainer; the flag is exposed for custom datasets that mix masked and unmasked rows.

## References

- SFT as next-token cross-entropy is the standard language-model pretraining/finetuning objective (Vaswani et al., 2017; Radford et al., 2018/2019).
- Implementation: `vendor/mlx-lm-lora/mlx_lm_lora/trainer/sft_trainer.py` (`default_loss`).

## See also

- [Algorithm Guide](Algorithm-Guide) · [Train](Train) · [DPO](DPO) · [LoRA](LoRA) · [QAT](QAT)