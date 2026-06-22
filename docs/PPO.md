# PPO

**Family:** Reinforcement / online · **Reference model:** yes · **Judge:** LLM or human · **QAT:** no

> The textbook clipped policy-gradient objective applied to LMs — the most expressive and the most finicky loop.

## Overview

**PPO (Schulman et al., 2017) as applied to LMs** is the textbook clipped policy optimisation. For each prompt the trainer samples two completions, asks the judge which is better, and treats them as `(chosen, rejected)`. It then computes log-ratios against the reference model and minimises the clipped surrogate objective on both sequences, plus a KL penalty.

It is the most powerful and the most finicky of the loops: the `epsilon` clip range, the `beta` KL weight, and the judge quality all matter a lot.

## Intuition

- The objective is the standard PPO surrogate: `−min(ρ · A, clip(ρ, 1−ε, 1+ε) · A)`, with `ρ = π / π_ref` and `A` derived from the chosen-rejected reward gap.
- The clip range `epsilon = 0.2` is the classic Schulman default; tightening it (e.g. 0.1) makes updates more conservative, loosening it (e.g. 0.3) lets the policy move further per step.
- KL is added on top of the surrogate as a regulariser, not as part of the advantage.

## Objective (math)

For a prompt, two completions, and a winner `w`. The advantage is *not* a reward-model output here — it is derived from the log-prob gap between chosen and rejected, after the judge has decided which is which.

```text
A           =  log π_θ(y_c)  −  log π_θ(y_r)                    (per-sequence advantage)
A_norm      =  ( A  −  mean )  /  ( std + 1e−8 )

ρ_c         =  exp( log π_θ(y_c)  −  log π_ref(y_c) )
ρ_r         =  exp( log π_θ(y_r)  −  log π_ref(y_r) )

ℒ_surr      =  − min( ρ_c · A_norm,  clip( ρ_c, 1−ε, 1+ε ) · A_norm )
             =  − min( ρ_r · (−A_norm),  clip( ρ_r, 1−ε, 1+ε ) · (−A_norm) )

ℒ_PPO       =  ℒ_surr  +  β · ( mean( log π_θ  −  log π_ref ) )
```

## Dataset format

Same as the other online loops: only a `prompt` field at training time. Completions are sampled from the policy and the chosen/rejected split comes from the judge.

The bundled default is `mlx-community/Human-Like-DPO`.

## When to use it

The classic, the most expressive, and the most finicky. Reach for PPO when the other loops are under-performing on a metric the judge captures well, and you have time to tune `epsilon` and `beta` together. Always log `clip_fraction` — if it sits at 0% the policy is not moving, if it sits at >30% the policy is moving too far per step.

## PPO-specific settings

In addition to the [shared SFT substrate](SFT#shared-configuration-reference):

| Setting | Default | What it actually changes |
|---|---|---|
| `beta` | `0.1` | KL regulariser weight, added to the clipped surrogate. |
| `epsilon` | `0.2` | PPO clip range for the importance ratio. The classic value; lower it for more conservative updates. |
| `dpo_cpo_loss_type` | `sigmoid` | Loss variant (the chosen/rejected split is the same; only the inner objective changes — the runner passes it through). |
| `delta` | `50.0` | Drift-penalty coefficient for `dpop`. |
| `judge` | `Qwen/Qwen3-0.6B` | Pairwise judge (LLM or `human`). |
| `judge_system` | `—` | Rubric system prompt for the judge. |
| `max_completion_length` | `512` | Maximum sampled completion length. |
| `temperature` | `0.8` | Sampling temperature for the policy completions. |
| `reference_model_path` | `—` | Frozen reference used in the importance ratio. |

## In the app

On the **Train** tab, PPO shows an **Online Preference** block on top of the shared form:

- **Completion** → `max_completion_length`; **Temp** → `temperature`; **Epsilon** → `epsilon` (the PPO clip range; classic default `0.2`).
- **Judge** picker (segmented LLM / User): LLM → `judge` (model id / local path); User → `judge = "human"`.
- **Judge model** field (shown when Judge = LLM).
- **Judge system prompt** → `judge_system` (shown for the LLM judge — the rubric PPO will amplify, so make it precise).
- **Dataset Columns** — Prompt.

Not exposed in the UI for online modes (use app defaults; edit via YAML): `beta` (`0.1`, KL regulariser weight), `dpo_cpo_loss_type` (`sigmoid`), `delta` (`50.0`), `reference_model_path` (empty ⇒ second copy of the base model, used in the importance ratio).

Shared form: Model & Data; Fine-tune (LoRA/DoRA/Full + Quantization); Training Settings; Output. (QAT is not applicable to online loops.)

## Tips & gotchas

- Log `clip_fraction`. If it is consistently > 0.3, the policy is moving too far per step — either lower `lr` or tighten `epsilon` to 0.1.
- `epsilon = 0.2` is the original PPO default and is a fine starting point; do not lower it until you have seen the policy train for at least one full pass.
- A KL weight (`beta`) that is too small lets the policy drift far from the reference; a `beta` that is too large suppresses the policy before it learns anything. Start at 0.1 and adjust based on the `kl_penalty` log.
- The judge matters more than the algorithm. PPO amplifies whatever the judge rewards, for better or worse.

## References

- Schulman et al., 2017, *Proximal Policy Optimization Algorithms*.
- Implementation: `vendor/mlx-lm-lora/mlx_lm_lora/trainer/ppo_trainer.py`.

## See also

- [GRPO](GRPO) · [Online-DPO](Online-DPO) · [RLHF-Reinforce](RLHF-Reinforce) · [Algorithm Guide](Algorithm-Guide)