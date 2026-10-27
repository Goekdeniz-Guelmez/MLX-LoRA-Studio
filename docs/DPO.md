# Direct Preference Optimisation (DPO)

**Family:** Preference · **Reference model:** yes · **Judge:** no · **QAT:** yes

> Train directly on human preference pairs with a closed-form loss — no reward model, no sampling loop.

## Overview

**Direct Preference Optimisation (Rafailov et al., 2023)** trains the model directly on human preference pairs `(chosen, rejected)` for the same prompt. There is no reward model and no sampling loop — just a closed-form loss that pulls the policy's log-probability of the chosen completion up and the rejected completion down, regularised by a frozen **reference model** so the policy does not drift.

The reference is normally the base model you started from. Loading a different one (e.g. an instruction-tuned SFT checkpoint) shifts the implicit reward baseline and is a common knob for steering the resulting behaviour.

## Intuition

- DPO is derived from the closed-form solution of the KL-constrained RL objective, so the loss is mathematically equivalent to RLHF with a learned reward model — but you skip the reward model entirely.
- The implicit reward is `β · log(π_θ(chosen) / π_ref(chosen)) − β · log(π_θ(rejected) / π_ref(rejected))`. A larger `β` makes the loss more aggressive, a smaller one softer.
- `loss_type = "sigmoid"` is the original DPO; `hinge` is a margin-style loss; `ipo` regularises toward a constant target (more robust to noise); `dpop` adds an explicit reference-drift penalty scaled by `delta`.

## Objective (math)

For a single preference pair `(y_c, y_r)` and prompt `x`. `β` is the temperature: higher `β` pushes the policy harder toward the implicit reward. `delta` is only used in `dpop` and controls how much drift from the reference is penalised.

```text
logits  =  ( log π_θ(y_c|x)  −  log π_θ(y_r|x) )
         −  ( log π_ref(y_c|x) −  log π_ref(y_r|x) )

ℒ_DPO   =  − log σ( β · logits )                                  (sigmoid)
         =  max( 0,  1 − β · logits )                             (hinge)
         =  ( logits − 1/(2β) )²                                  (ipo)
         =  − log σ( β · logits )
            + δ · max( 0,  log π_ref(y_c|x) − log π_θ(y_c|x) )    (dpop)
```

## Dataset format

DPO expects a preference dataset with one row per preference, containing at minimum `chosen` and `rejected`. The `prompt` column is optional but recommended; if present it is prepended to both completions before tokenisation.

The bundled default is `mlx-community/Human-Like-DPO`, which has the prompt/chosen/rejected shape.

## When to use it

After SFT, when you have a static preference dataset (UltraFeedback, HelpSteer, Anthropic HH). DPO is the cheapest preference algorithm — one forward pass per completion on the policy, one forward pass on the reference.

Use `loss_type=ipo` if you have noisy or contradictory labels, `loss_type=dpop` if the policy starts drifting from the reference in ways the SFT loss did not catch.

## DPO-specific settings

In addition to the [shared SFT substrate](SFT#shared-configuration-reference):

| Setting | Default | What it actually changes |
|---|---|---|
| `beta` | `0.1` | Temperature inside the DPO sigmoid. Lower = softer updates, higher = more aggressive divergence from the reference. |
| `dpo_cpo_loss_type` | `sigmoid` | `sigmoid` (vanilla DPO), `hinge` (margin), `ipo` (squared error around `1/(2β)`), `dpop` (reference-drift penalty). |
| `delta` | `50.0` | Coefficient on the `dpop` reference-drift penalty. Ignored unless `dpo_cpo_loss_type = dpop`. |
| `reference_model_path` | `—` | Path/HF id of the frozen reference. If empty, the trainer instantiates a second copy of the base model (doubles GPU memory). |
| `prompt_feature` / `chosen_feature` / `rejected_feature` | `auto` | Column-name overrides for the preference schema. |

## In the app

On the **Train** tab, DPO shows a **Preference And Judge** block on top of the shared form:

- **Beta** → `beta`; **Delta** → `delta` (used only for `dpop`); **Loss** picker (Sigmoid / Hinge / IPO / DPOP) → `dpo_cpo_loss_type`; **Reference model path** → `reference_model_path` (empty ⇒ second copy of the base model in memory).
- **Dataset Columns** — Chosen, Rejected, Prompt (column-name overrides).

Shared form: Model & Data; Fine-tune (LoRA/DoRA/Full + Quantization); Training Settings (LR, Optimizer, schedule, batch, iters, grad accumulation, etc.); Output; **QAT** (DPO supports QAT).

## Tips & gotchas

- Set `reference_model_path` to the SFT checkpoint, not the original base model — the implicit reward is then `Δ log-prob vs the SFT model`, which is what your labels reflect.
- Watch `accuracies` and `margins` in the live metrics. `accuracies > 0.7` with `margins > 0` means the loss is doing real work; if `margins` plateaus, raise `beta` slightly.
- If you see NaNs, the most common cause is `loss_type=dpop` with `delta` too large for the current `lr`. Drop `delta` to 10 first.
- `efficient_long_context` applies here too — preference datasets with long answers benefit from chunked forward passes.

## References

- [Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290).
- Implementation: `vendor/mlx-lm-lora/mlx_lm_lora/trainer/dpo_trainer.py`.

## See also

- [CPO](CPO) · [ORPO](ORPO) · [Online-DPO](Online-DPO) · [SFT](SFT) · [Algorithm Guide](Algorithm-Guide)