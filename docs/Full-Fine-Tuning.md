# Full Fine-Tuning

**Adaptation method** · `train_type: full`

> Unfreeze every parameter. Textbook fine-tuning — highest capacity, highest memory, rarely viable past 1B on a laptop.

## Overview

**`full` fine-tuning** unfreezes the entire model. Every weight is trainable, the optimiser state is proportional to the parameter count, and a 7B model needs ~28 GB of activation + optimiser memory at fp16. On a laptop this is rarely viable past the 1B scale.

All three adaptation methods share the same forward pass; only the parameter list passed to `nn.value_and_grad(model, …)` changes. That is why every algorithm works with every adaptation method.

## Intuition

- Full fine-tuning is the textbook case. The optimiser keeps a state tensor per parameter (two for Adam-family), so memory scales with the full parameter count rather than a low-rank slice.
- Switching to `full` removes the adapter wrapper entirely; `num_layers` is ignored.

## Objective (math)

```text
y  =  W · x            (every W is trainable)
```

## What the settings change

Set `train_type: full`. The `lora_parameters.*` and `num_layers` settings are ignored. The [shared SFT substrate](SFT#shared-configuration-reference) otherwise applies; `fuse` has no adapter to merge.

## When to use it

Reserved for cases where LoRA / DoRA are not expressive enough — e.g. training a small model from scratch on a domain corpus. On a 16 GB Apple Silicon machine the practical limit is ~1B parameters at fp16.

## Memory expectations

Approximate, single-GPU Apple Silicon, `batch_size 2`, 2048-token sequences (from the README):

| Base model size | Full FT |
|---|---|
| 1B–3B | 12 GB |
| 7B–8B | 32 GB |
| 13B | 56 GB |
| 70B | — |

## In the app

On the **Train** tab → **Fine-tune** section: pick **FULL** in the segmented Fine-tune picker (LORA / DORA / FULL) → `train_type: full`. The **LoRA Settings** block (Layers / Rank / Scale / Dropout) is hidden — those knobs are not meaningful when every weight is trainable. The **Quantization** picker remains available. In **Training Settings**, enable **grad checkpoint** from the start and use a low Learning Rate (1e-6 to 5e-6 with AdamW).

## Tips & gotchas

- For `full` fine-tuning, set `grad_checkpoint = true` from the start; the activation memory alone is enough to OOM a 7B model on a laptop.
- QAT has no effect on a `full` run — it only changes how LoRA/DoRA adapters are trained. If you are doing `train_type=full`, leave `qat_enable = false`.
- Use a low learning rate: full fine-tuning usually wants `1e-6` to `5e-6` with AdamW.

## See also

- [LoRA](LoRA) · [DoRA](DoRA) · [QLoRA](QLoRA) · [Optimizers](Optimizers) · [SFT](SFT)