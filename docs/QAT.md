# Quantization-Aware Training (QAT)

**In-training quantization** · `qat_enable: true`

> Install a straight-through-estimator fake-quantise hook on every `nn.Linear` so the adapter trains *as if* it will be deployed quantised. Keeps outlier channels tame; only affects LoRA/DoRA adapters.

## Overview

`mlx-lm-lora` supports **two distinct kinds of quantisation** that are easy to confuse. This page covers **Quantization-Aware Training (QAT)**. (See [QLoRA](QLoRA) for the load-time variant.)

QAT is a small hook installed on every `nn.Linear` after the first optimiser step. The hook fake-quantises the weight on the way *into* the forward pass (straight-through estimator), so the model trains as if it would be quantised at inference time. The optimiser still sees and updates the full-precision weights, so the gradient is unaffected.

It composes with load-time quantisation: a typical "QLoRA + QAT" run loads the base model in 4-bit and then trains with the QAT hook enabled so the LoRA updates are robust to that 4-bit precision. QAT is only effective for the SFT/DPO/ORPO trainers (the others do not call `_install_qat_hooks`).

## Intuition

- QAT is the difference between a model that *was* trained in fp16 and *deployed* in 4-bit (which loses accuracy on outlier channels) and a model that was trained *as if* it would be deployed in 4-bit (which learns to keep the outliers tame).
- The straight-through estimator (STE) is the only "trick" in QAT: the forward uses the quantised weight and the backward is the identity, so gradients flow through unchanged. Without the STE, the quantise-then-round step would be zero almost everywhere and gradients would vanish.
- The QAT hook is enabled after the first optimiser step (`qat_start_step`) and re-applied every `qat_interval` steps. That deferred start is important: the very first optimiser step on a freshly initialised LoRA would quantise noise.

## Objective (math)

**Symmetric fake quantise** (applied inside the forward):

```text
# Same arithmetic as load-time, but at runtime on the current weight tensor:
Ŵ   =  s · clip( round( W / s ),  q_min,  q_max )

# The forward uses Ŵ; the backward uses an STE:
∂ℒ / ∂W   =  ∂ℒ / ∂Ŵ                       (identity — gradient flows through)
```

The hook is implemented as:

```text
self.weight  =  w  +  stop_gradient( quantize(w)  −  w )     # forward sees Ŵ
out          =  original_forward(self, x)
self.weight  =  w                                             # restore for optimiser
```

The `stop_gradient` around `( quantize(w) − w )` is the STE. The `+ w` outside it means the forward value is exactly `Ŵ`, the backward value is 1, and the optimiser only ever touches the full-precision `w`.

## What the settings change

| Setting | Default | What it actually changes |
|---|---|---|
| `qat_enable` | `false` | Install the STE fake-quantise hook on every `nn.Linear` after the first optimiser step. Only effective for SFT/DPO/ORPO. |
| `qat_bits` | `8` | Bit-width used by the hook. Match the inference quantisation: deploy at 4-bit ⇒ `qat_bits=4`; deploy at 8-bit ⇒ `qat_bits=8`. |
| `qat_group_size` | `64` | Group size used by the hook. 0 or negative = per-tensor. Match the deployment group size; 64 or 128 are common. |
| `qat_start_step` | `1` | First optimiser step on which to install the hook. Set higher if your first few steps see NaN gradients. |
| `qat_interval` | `1` | Re-apply the QAT projection every N steps. Default projects every step; raise to e.g. `4` if projection shows up in your profile. |

## Which to pick

- **4-bit load + LoRA + QAT (8-bit):** the "QLoRA" recipe. The QAT-on-8-bit hook makes the adapter robust to being merged back into a 4-bit base.
- **8-bit load + LoRA + QAT (8-bit):** gentler inference quantisation, more stable loss curve.
- **No load quantisation + QAT (4-bit):** useful when you want the *base* model to stay in fp16 but the *adapter* to be 4-bit-deployable. QAT is the only way to get a 4-bit adapter that does not lose accuracy on outlier channels.

## In the app

On the **Train** tab, the **QAT** section is a toggle plus four fields:

- **Enable** toggle → `qat_enable`.
- **Bits** → `qat_bits`; **Group** → `qat_group_size` (0 = per-tensor); **Start** → `qat_start_step`; **Interval** → `qat_interval`.

The section is only effective for SFT, DPO, and ORPO (the only trainers that install the hook). For the "QLoRA + QAT" recipe, set **Quantization** to 4-bit in the Fine-tune section and enable QAT here.

## Tips & gotchas

- QAT does not help the *base* model — it only changes how the LoRA/DoRA adapters are trained. If you are doing `train_type=full`, QAT has no effect on the merged output.
- If the loss starts to oscillate or NaN after a few hundred steps, the most common cause is QAT with a `qat_bits` too aggressive for the current learning rate. Drop `qat_bits` to 8 and try again.
- Group size 32 is the most sensitive to outliers; 128 is the most forgiving. When in doubt, raise the group size before lowering the bit-width.

## References

- QAT with the straight-through estimator (Bengio et al., 2013, *Estimating or Propagating Gradients Through Stochastic Neurons*); the affine scheme mirrors `mlx.nn.quantize`.
- Implementation: `_install_qat_hooks` in `vendor/mlx-lm-lora/mlx_lm_lora/trainer/sft_trainer.py`.

## See also

- [QLoRA](QLoRA) · [LoRA](LoRA) · [DoRA](DoRA) · [SFT](SFT) · [DPO](DPO) · [ORPO](ORPO)