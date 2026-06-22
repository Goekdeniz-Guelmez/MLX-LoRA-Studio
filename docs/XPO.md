# XPO (Exploratory Preference Optimisation)

**Family:** Reinforcement / online · **Reference model:** yes · **Judge:** LLM or human · **QAT:** no

> Online DPO plus an explicit exploration bonus proportional to the KL from the reference — rewards trying completions that differ from the reference.

## Overview

**XPO (Exploratory Preference Optimisation)** is the online-DPO family plus an explicit **exploration bonus** proportional to the KL divergence between the current policy and the reference. Setting `alpha > 0` rewards the model for trying completions that are different from the reference, which mitigates the "stuck on the reference completion" pathology of plain online DPO. `alpha` can be a single float or a list (one per epoch) so the bonus can decay over training.

## Intuition

- XPO's bonus term `alpha · (KL(policy ‖ ref) on chosen + KL on rejected)` rewards moving away from the reference. It is the opposite sign of a KL penalty.
- The result is a model that is incentivised to *try new things* early in training and to settle into a stable region as the bonus decays.
- Use a single `alpha` for constant exploration, or pass a list to decay it epoch by epoch.

## Objective (math)

XPO loss is DPO with an additive exploration bonus proportional to the KL. A positive `alpha` means *reward me for being different from the reference*. When `alpha` is a list, `get_current_alpha(step, total, schedule)` returns the schedule element for the current step (one entry per epoch, with the last entry held to the end).

```text
bonus    =  α · ( KL(π_θ ∥ π_ref) on chosen  +  KL(π_θ ∥ π_ref) on rejected )

ℒ_XPO    =  ℒ_DPO  −  bonus
```

## Dataset format

Same as [Online DPO](Online-DPO#dataset-format): only a `prompt` field at training time; completions are sampled and judged on the fly.

## When to use it

When online DPO collapses toward the reference completion and you want a controlled amount of exploration. Try `alpha=1e-4` first, then either hold it constant or schedule it down across epochs.

## XPO-specific settings

In addition to the [shared SFT substrate](SFT#shared-configuration-reference):

| Setting | Default | What it actually changes |
|---|---|---|
| `beta` | `0.1` | DPO temperature. |
| `alpha` | `1e-5` | Exploration-bonus coefficient. Single float = constant. A list (e.g. `[1e-4, 1e-5, 1e-6]`) gives a per-epoch schedule that decays. |
| `dpo_cpo_loss_type` | `sigmoid` | Base DPO loss variant. |
| `delta` | `50.0` | Drift-penalty coefficient for `dpop`. |
| `judge` | `Qwen/Qwen3-0.6B` | LLM or `human` for the pairwise judge. |
| `judge_system` | `—` | Rubric system prompt for the judge. |
| `max_completion_length` | `512` | Maximum sampled completion length. |
| `temperature` | `0.8` | Sampling temperature for the policy. |
| `reference_model_path` | `—` | Path/HF id of the frozen reference model. |

## In the app

On the **Train** tab, XPO shows an **Online Preference** block on top of the shared form:

- **Completion** → `max_completion_length`; **Temp** → `temperature`.
- **Alpha schedule** → `alpha` (a single float for constant exploration, or a list like `[1e-4, 1e-5, 1e-6]` for a per-epoch decay).
- **Judge** picker (segmented LLM / User): LLM → `judge` (model id / local path); User → `judge = "human"`.
- **Judge model** field (shown when Judge = LLM).
- **Judge system prompt** → `judge_system` (shown for the LLM judge).
- **Dataset Columns** — Prompt.

Not exposed in the UI for online modes (use app defaults; edit via YAML): `beta` (`0.1`), `dpo_cpo_loss_type` (`sigmoid`), `delta` (`50.0`), `reference_model_path` (empty ⇒ second copy of the base model).

Shared form: Model & Data; Fine-tune (LoRA/DoRA/Full + Quantization); Training Settings; Output. (QAT is not applicable to online loops.)

## Tips & gotchas

- Default `alpha = 1e-5` is very small. If completions look indistinguishable from the reference, raise `alpha` to `1e-4` and watch `exploration_bonus` in the live metrics.
- For a 3-epoch run, try `alpha = [1e-4, 1e-5, 1e-6]` — strong exploration early, decaying to nothing by the final epoch.

## References

- XPO builds on Online DPO (and thus DPO, Rafailov et al., 2023) with an additive KL exploration bonus.
- Implementation: `vendor/mlx-lm-lora/mlx_lm_lora/trainer/xpo_trainer.py`.

## See also

- [Online-DPO](Online-DPO) · [DPO](DPO) · [GRPO](GRPO) · [Algorithm Guide](Algorithm-Guide)