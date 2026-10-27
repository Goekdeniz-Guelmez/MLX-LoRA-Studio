# LoRA (Low-Rank Adaptation)

**Adaptation method** · `train_type: lora`

> Keep the base model frozen; learn a low-rank update `ΔW = (α/r) · B · A` on each targeted linear layer. ~4M trainable params on a 7B model.

## Overview

Every trainer in `mlx-lm-lora` operates on the same base model loaded by `mlx_lm_lora.utils.from_pretrained`. What changes between `lora`, `dora`, and `full` is which tensors are **trainable** and how the gradients are computed. **LoRA** keeps the base model frozen and learns a low-rank update to each targeted linear layer.

LoRA is the default adaptation method in MLX LoRA Studio and the right choice for almost every run.

## Intuition

- LoRA pretends the optimal weight change `ΔW` is rank-deficient: `ΔW = (α/r) · B · A` where `A ∈ R^{r×in}`, `B ∈ R^{out×r}`, `r ≪ min(in, out)`. For a 7B model with `r=8` this drops trainable parameters from ~7B to ~4M.
- All three adaptation methods share the same forward pass; only the parameter list passed to `nn.value_and_grad(model, …)` changes. That is why every algorithm works with every adaptation method.

## Objective (math)

Let `W₀ ∈ R^{out×in}` be the frozen base weight, `x` the input and `y` the output of the targeted `nn.Linear` layer.

```text
A    ~  𝒩( 0, σ² )           (initialised)
B    =  0                     (initialised — first step is a no-op)
ΔW   =  ( α / r )  ·  B · A
y    =  ( W₀ + ΔW ) · x  +  dropout( ΔW · x )       (if dropout > 0)
```

The `α / r` ratio is what the original paper called the *scaling factor*; in `mlx-lm-lora` the same number is split into the `scale` (α) and `rank` (r) settings.

## What the settings change

| Setting | Default | What it actually changes |
|---|---|---|
| `train_type` | `lora` | Pick `lora`, `dora`, or `full`. Switching to `full` removes the adapter wrapper entirely. |
| `lora_parameters.rank` | `8` | Inner rank of the low-rank update. 4 = tiny, 8 = typical, 16 = high capacity, 32+ = usually overkill on a laptop. |
| `lora_parameters.scale` | `20.0` | Magnitude scaling (α). Effective update is `(α/r) · B·A`, so the ratio to `rank` is what matters. Reference defaults `(rank=8, scale=20.0)` for a 2.5× ratio. |
| `lora_parameters.dropout` | `0.0` | Dropout applied to `ΔW · x` before it is added to `W₀ · x`. Only > 0 for small datasets or visible overfitting. |
| `num_layers` | `16` | How many of the top transformer layers receive LoRA adapters. The trainer counts from the top, so `num_layers = 8` on a 32-layer model targets layers 24–31. |
| `resume_adapter_file` | `—` | Path to a saved `adapters.safetensors`. Works for any `train_type` (the file just needs to match the layer names of the new run). |
| `fuse` | `true` | If true, the LoRA updates are merged back into `W₀` after training and the merged model is saved to `adapter_path`. Disable to keep adapter files separate. |

## When to use it

LoRA is the right default for everything in this app. ~4M trainable parameters, fits in CPU RAM, fast iteration.

## In the app

On the **Train** tab → **Fine-tune** section:

- **Fine-tune** picker (segmented): LORA / DORA / FULL → `train_type`.
- With LORA selected, **LoRA Settings**: **Layers** → `num_layers`, **Rank** → `rank`, **Scale** → `scale`, **Dropout** → `dropout`.
- **Quantization** picker: None / 4-bit / 6-bit / 8-bit / MXFP4 (see [QLoRA](QLoRA)).
- Resume a previous adapter via the **Runs** tab → **Open** (loads a run back into the form, including `resume_adapter_file`), or set the run folder in **Output**.
- `fuse` (merge adapters back into the base after training) is a toggle in **Training Settings**.

## Tips & gotchas

- Start with `rank=8, scale=20.0, dropout=0.0`. If the loss plateaus, raise `rank` to 16 *before* changing the learning rate — capacity is usually the bottleneck, not step size.
- If you change `train_type` between runs, delete the old `adapters.safetensors` first — the layer name conventions differ and a stale file will silently fail to load.

## References

- [Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685).
- Implementation: `mlx_lm_lora.utils.from_pretrained` → `linear_to_lora_layers`.

## See also

- [DoRA](DoRA) · [Full-Fine-Tuning](Full-Fine-Tuning) · [QLoRA](QLoRA) · [Optimizers](Optimizers) · [SFT](SFT)