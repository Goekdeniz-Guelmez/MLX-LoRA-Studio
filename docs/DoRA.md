# DoRA (Weight-Decomposed Low-Rank Adaptation)

**Adaptation method** · `train_type: dora`

> LoRA plus a magnitude–direction decomposition: the combined weight is split into a unit-norm direction and a learnable magnitude, tuned independently. Matches full fine-tuning more closely on instruction-following.

## Overview

**DoRA** keeps the same low-rank LoRA update but factorises the combined weight into a magnitude vector and a direction matrix so the two can be tuned independently. It costs roughly the same memory and time as LoRA; the only downside is a slightly larger adapter file.

## Intuition

- DoRA decomposes the (frozen + LoRA) weight into a unit-norm direction `V` and a learnable magnitude `m`, then writes `W' = m · V / ‖V‖`. The motivation is empirical: full fine-tuning tends to update direction and magnitude by very different amounts, and DoRA reproduces that behaviour while keeping the parameter count near LoRA's.
- All three adaptation methods share the same forward pass; only the parameter list passed to `nn.value_and_grad(model, …)` changes.

## Objective (math)

Let `W₀ ∈ R^{out×in}` be the frozen base weight.

```text
W'        =  W₀  +  ( α / r )  ·  B · A
V         =  W'                              (frozen after each step)
m         =  ‖W₀‖_c                          (per-column magnitude, learnable)
W_dora    =  m  ·  V  /  ‖V‖
y         =  W_dora · x
```

The unit-norm rescaling on `V` is what makes DoRA different from *LoRA plus a magnitude multiplier*.

## What the settings change

DoRA shares the [LoRA settings table](LoRA#what-the-settings-change) (`rank`, `scale`, `dropout`, `num_layers`, `fuse`, `resume_adapter_file`); set `train_type: dora` to select the DoRA wrapper.

## When to use it

DoRA is worth trying when LoRA plateaus on a metric that tracks style or format compliance (DoRA is reported to match full fine-tuning more closely on instruction-following). It costs roughly the same memory and time as LoRA; the only downside is a slightly larger adapter file.

## In the app

On the **Train** tab → **Fine-tune** section: pick **DORA** in the segmented Fine-tune picker (LORA / DORA / FULL) → `train_type: dora`. The same **LoRA Settings** controls apply to DoRA — **Layers** (`num_layers`), **Rank**, **Scale**, **Dropout** — alongside the **Quantization** picker. `fuse` is a toggle in **Training Settings**.

## Tips & gotchas

- DoRA's magnitude vector is per-output-column, so a layer with a large output dim stores more DoRA parameters than LoRA at the same rank. The difference is small (a few hundred floats per layer) but it shows up in the adapter file size.
- If you change `train_type` between runs, delete the old `adapters.safetensors` first — the layer name conventions differ and a stale file will silently fail to load.

## References

- Liu et al., 2024, *DoRA: Weight-Decomposed Low-Rank Adaptation*.
- Implementation: `mlx_lm_lora.utils.from_pretrained` → DoRA wrapper.

## See also

- [LoRA](LoRA) · [Full-Fine-Tuning](Full-Fine-Tuning) · [QLoRA](QLoRA) · [Optimizers](Optimizers)