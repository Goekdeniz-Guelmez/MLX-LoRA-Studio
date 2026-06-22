# QLoRA (Load-Time Quantization)

**Quantization** · `load_in_4bits` / `load_in_6bits` / `load_in_8bits` / `load_in_mxfp4`

> Load the base model in N-bit integers with a per-group scale, then train full-precision LoRA/DoRA adapters on top. The standard "QLoRA" recipe for fitting 7B–13B models on a 16 GB laptop.

## Overview

`mlx-lm-lora` supports **two distinct kinds of quantisation** that are easy to confuse. This page covers **load-time quantisation**. (See [QAT](QAT) for the in-training variant.)

When the base model is read from disk, `mlx.nn.quantize` replaces each `nn.Linear` with a `QuantizedLinear` that stores the weights in N-bit signed integers and a per-group scale. The base model stays in memory in that form for the rest of the run, and the LoRA / DoRA adapters are kept in full precision on top of it. Quantisation happens at load time and is irreversible for the loaded model object — the only way to change the base precision is to re-load the model (restart the runner after changing the `load_in_*` flags).

Load-time quantisation composes with QAT: a typical "QLoRA + QAT" run loads the base model in 4-bit and then trains with the QAT hook enabled so the LoRA updates are robust to that 4-bit precision.

## Intuition

The matrix `W ∈ R^{out×in}` is split along the last axis into groups of `group_size` consecutive columns, each group is divided by a scale `s_g = max(|W_g|) / qmax`, and the values are rounded to signed N-bit integers. The quantised weight is `Q_g = round(W_g / s_g)` and the dequantised weight is `Q_g · s_g ≈ W_g`.

## Objective (math)

Affine quantisation (all four modes):

```text
# Per group of group_size consecutive columns of W:
s_g   =  max(| W_g |) / q_max                 (positive scale)
Q_g   =  clip( round( W_g / s_g ),  q_min,  q_max )    (N-bit signed int)
Ŵ_g   =  Q_g  ·  s_g                           (dequantised)
```

with `q_max = 2^(N−1) − 1` and `q_min = −q_max − 1`. For `N = 4`, `q_max = 7` and `q_min = −8`. For **MXFP4** the format is identical but the group size is fixed at 32 and the scale is stored as an 8-bit E8M0 value (a *microscaling* format).

## What the settings change

| Setting | Default | What it actually changes |
|---|---|---|
| `load_in_4bits` | `false` | Quantise the base model to 4-bit on load (group size 128). Most aggressive; pairs naturally with LoRA/DoRA. |
| `load_in_6bits` | `false` | 6-bit quantisation (group size 128). Rarely the right choice — usually 4-bit or 8-bit wins. |
| `load_in_8bits` | `false` | 8-bit quantisation (group size 128). Safe default for preference training and long contexts with precise numerics. |
| `load_in_mxfp4` | `false` | MXFP4 (microscaling 4-bit), group size 32, 8-bit E8M0 scale. Functionally similar to 4-bit on Apple Silicon but with smaller groups. |

## Memory expectations

Approximate, single-GPU Apple Silicon, `batch_size 2`, 2048-token sequences (from the README):

| Base model size | QLoRA 4-bit |
|---|---|
| 1B–3B | 5 GB |
| 7B–8B | 7 GB |
| 13B | 10 GB |
| 70B | 38 GB |

## Which to pick

- **4-bit load + LoRA + QAT (8-bit):** the "QLoRA" recipe. Smallest memory footprint, suitable for 7B–13B models on a 16 GB laptop. The QAT-on-8-bit hook makes the adapter robust to being merged back into a 4-bit base.
- **8-bit load + LoRA:** slightly higher memory than 4-bit, but the inference quantisation is gentler and the loss curve is more stable. The safe default for preference training.
- **MXFP4:** only choose this if your deployment target uses MXFP4. On Apple Silicon the throughput difference vs. regular 4-bit is small.
- **6-bit:** rarely the right call — pick 4-bit (smallest) or 8-bit (highest quality).

## In the app

On the **Train** tab → **Fine-tune** section, the **Quantization** dropdown selects the load-time precision:

- **None** / **4-bit** / **6-bit** / **8-bit** / **MXFP4** → `load_in_4bits` / `load_in_6bits` / `load_in_8bits` / `load_in_mxfp4` (group size 128, or 32 for MXFP4).

Pair it with LoRA or DoRA in the same Fine-tune section. Quantisation is applied at load time and is irreversible for the loaded model object — restart the run after changing it. For the "QLoRA + QAT" recipe, also enable the [QAT](QAT) section.

## Tips & gotchas

- Quantisation is irreversible for the loaded model object — restart the runner after changing the `load_in_*` flags.
- Group size 32 (the MXFP4 default) is the most sensitive to outliers; 128 is the most forgiving. When in doubt, raise the group size before lowering the bit-width.

## References

- Dettmers et al., 2023, *QLoRA: Efficient Finetuning of Quantized LLMs*; the load-time affine scheme mirrors `mlx.nn.quantize`.
- Microscaling (MXFP4) format: NVIDIA / OCP Microscaling Formats specification.

## See also

- [QAT](QAT) · [LoRA](LoRA) · [DoRA](DoRA) · [Full-Fine-Tuning](Full-Fine-Tuning) · [SFT](SFT)