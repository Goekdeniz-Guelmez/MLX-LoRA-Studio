# ORPO

**Family:** Preference · **Reference model:** no · **Judge:** no · **QAT:** yes

> Fold SFT and preference tuning into a single loss — no reference model, no SFT warm-up stage.

## Overview

**ORPO (Hong et al., 2024)** folds the SFT cross-entropy and the odds-ratio preference term into a single loss. There is no reference model and no SFT warm-up step — the chosen response is pushed up and the rejected one pulled down by the same gradient that improves the model's next-token log-likelihood. In `mlx_lm_lora` it accepts an optional `preference_score` per example so heterogeneous preference strengths (e.g. UltraFeedback-style 0..10 scores) can be used as a soft target.

## Intuition

- ORPO's signature trick is the log-odds term: `log σ( log π(chosen) − log π(rejected) )`. The same gradient that lowers NLL on the chosen completion also increases that gap.
- `preference_score` (default 1.0) lets the trainer scale the chosen log-prob per example, so a soft preference of 0.3 still gets a smaller push than a hard preference of 1.0.
- `reward_scaling` is accepted but the upstream implementation does not actually use it as a separate multiplier — it is reserved for future variants.

## Objective (math)

The same loss reduces NLL on the chosen completion because the chosen log-prob term appears in both halves of the gradient. The optional `preference_score` rescales the chosen log-prob before the subtraction, so a row with `preference_score = 0.3` contributes roughly a third of the gradient of a row with score 1.0.

```text
log_odds  =  log π_θ(y_c|x)  −  log π_θ(y_r|x)        (mean over tokens)

ℒ_ORPO    =  − β · log σ( log_odds )
```

## Dataset format

ORPO requires `chosen` and `rejected` and **no** `prompt` column (the chosen and rejected strings are used verbatim, and the model is expected to learn the prompt–completion split on its own). Optionally a `preference_score` (float) column scales the per-example gradient.

The bundled default is `mlx-community/Josiefied-Qwen3-dpo-v1-flat`, a flattened DPO dataset.

## When to use it

When you want a single-stage alternative to "SFT then DPO". ORPO has been shown to work well on small chat models and is the natural choice for datasets that ship a per-example preference score (UltraFeedback-style `score_chosen - score_rejected`).

## ORPO-specific settings

In addition to the [shared SFT substrate](SFT#shared-configuration-reference):

| Setting | Default | What it actually changes |
|---|---|---|
| `beta` | `0.1` | Multiplier on the ORPO log-odds term. |
| `reward_scaling` | `1.0` | Reserved for future variants — the current ORPO loss does not use it as a separate multiplier. |
| `chosen_feature` / `rejected_feature` / `preference_score_feature` | `auto` | Column-name overrides. The ORPO trainer expects `chosen`, `rejected`, and optionally `preference_score`. |

## In the app

On the **Train** tab, ORPO shows a **Preference And Judge** block on top of the shared form:

- **Beta** → `beta`; **Delta** → `delta`; **Reward Scale** → `reward_scaling` (reserved — the current ORPO loss does not use it as a separate multiplier); **Loss** picker (Sigmoid / Hinge / IPO / DPOP) → `dpo_cpo_loss_type`.
- No **Reference model path** field — ORPO has no reference model and no SFT warm-up stage.
- **Dataset Columns** — Chosen, Rejected, **Preference score** (the optional per-example `preference_score` column).

Shared form: Model & Data; Fine-tune (LoRA/DoRA/Full + Quantization); Training Settings; Output; **QAT** (ORPO supports QAT).

## Tips & gotchas

- Verify your dataset does not have a `prompt` column — ORPO silently concatenates the prompt into the chosen/rejected if you let it, which is rarely what you want.
- `preference_score` should be normalised to roughly 0..1 before training; very large scores push the loss into saturated regions of `log_sigmoid`.
- ORPO does not need a separate SFT warm-up — the same loss improves the NLL. Skip a stage in your pipeline if you were planning "SFT then DPO".

## References

- Hong et al., 2024, *ORPO: Monolithic Preference Optimization without Reference Model*.
- Implementation: `vendor/mlx-lm-lora/mlx_lm_lora/trainer/orpo_trainer.py`.

## See also

- [DPO](DPO) · [CPO](CPO) · [SFT](SFT) · [Algorithm Guide](Algorithm-Guide)