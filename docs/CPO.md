# Contrastive Preference Optimisation (CPO)

**Family:** Preference · **Reference model:** no · **Judge:** no · **QAT:** yes

> DPO with the reference term dropped — faster, lighter, no second model in memory, at the cost of more sensitive `beta`/`delta` tuning.

## Overview

**CPO (Contrastive Preference Optimisation)** is DPO with the reference term dropped. The chosen-rejected log-prob gap is compared against an absolute target instead of a relative one — which means the policy can move further from the base model without a reference forward pass. It is faster to train and uses less memory, at the cost of being more sensitive to the `beta`/`delta` knobs.

## Intuition

- Without a reference, the policy can drift — the model is being told "chosen is better than rejected" but there is no anchor.
- Setting `loss_type = "dpop"` adds a hinge penalty `max(0, ref − π)` (CPO substitutes the policy log-prob for the reference in the drift penalty) to keep that drift bounded by `delta`.
- In practice CPO converges faster than DPO at the cost of needing slightly more careful `beta` and `delta` tuning.

## Objective (math)

Same shape as DPO without the reference term in the main loss. The `dpop` CPO variant substitutes the policy log-prob for the reference in the drift penalty, because the reference is not in the forward pass.

```text
logits  =  log π_θ(y_c|x)  −  log π_θ(y_r|x)

ℒ_CPO   =  − log σ( β · logits )                                  (sigmoid)
         =  max( 0,  1 − β · logits )                             (hinge)
         =  ( logits − 1/(2β) )²                                  (ipo)
         =  − log σ( β · logits )
            + δ · max( 0,  log π_θ(y_r|x) − log π_θ(y_c|x) )      (dpop)
```

## Dataset format

Same as DPO: one row per preference with `chosen` and `rejected`. CPO does not use the `prompt` column.

## When to use it

Same use case as DPO when you cannot afford a second reference model on the GPU, or when you want a more aggressive update. Pair with a slightly smaller `beta` than DPO would use.

## CPO-specific settings

In addition to the [shared SFT substrate](SFT#shared-configuration-reference):

| Setting | Default | What it actually changes |
|---|---|---|
| `beta` | `0.1` | Temperature inside the CPO sigmoid. |
| `dpo_cpo_loss_type` | `sigmoid` | Loss variant. CPO accepts the same four options as DPO; `dpop` here is the policy-side drift penalty. |
| `delta` | `50.0` | Coefficient for the CPO drift penalty. Only used with `loss_type = dpop`. |
| `chosen_feature` / `rejected_feature` | `auto` | Column-name overrides. CPO does not need a separate `prompt` column. |

## In the app

On the **Train** tab, CPO shows a **Preference And Judge** block on top of the shared form:

- **Beta** → `beta`; **Delta** → `delta` (used only for `dpop`); **Loss** picker (Sigmoid / Hinge / IPO / DPOP) → `dpo_cpo_loss_type`.
- No **Reference model path** field — CPO has no reference model, so it does not need a second copy of the base in memory.
- **Dataset Columns** — Chosen, Rejected (column-name overrides; CPO does not use a prompt column).

Shared form: Model & Data; Fine-tune (LoRA/DoRA/Full + Quantization); Training Settings; Output; **QAT** (CPO supports QAT).

## Tips & gotchas

- Use a smaller `beta` than DPO would use (start at 0.05) — without the reference term the gradient is unanchored and large `beta` overshoots.
- CPO is the only preference algorithm that does not need a second copy of the base model, so it is the right call when you are running on integrated Apple Silicon GPUs.
- If validation reward plateaus early, switch the loss to `dpop` and add a `delta` to keep drift bounded.

## References

- CPO shares its mathematical lineage with DPO (Rafailov et al., 2023); the reference-free variant is the contrastive/contrastive-direct line.
- Implementation: `vendor/mlx-lm-lora/mlx_lm_lora/trainer/cpo_trainer.py`.

## See also

- [DPO](DPO) · [ORPO](ORPO) · [SFT](SFT) · [Algorithm Guide](Algorithm-Guide)