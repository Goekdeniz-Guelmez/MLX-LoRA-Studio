# RLHF-REINFORCE

**Family:** Reinforcement / online · **Reference model:** optional · **Judge:** LLM · **QAT:** no

> The classic policy-gradient RLHF loop — conceptually simplest, highest variance, no clipping, no value head.

## Overview

**RLHF-REINFORCE** is the classic policy-gradient RLHF loop. The trainer samples completions from the policy, scores them with a scalar reward (an LLM judge, normally configured via the `judge` and `judge_system` settings), and applies a per-token REINFORCE objective regularised by an optional KL penalty against a reference model.

It is conceptually the simplest of the online algorithms (no clipping, no value head) but it has the highest variance, so it benefits from smaller learning rates and longer KL warm-up than DPO/PPO.

## Intuition

- The policy gradient is `−(reward − β · KL) · log π(action)`, summed over the sampled trajectory. There is no clipping and no value head, which is why the variance is high.
- In `mlx_lm_lora` the "reward" is a scalar produced by an LLM judge (the `judge` setting); there is no separate value model.
- KL is computed per token between the policy and the reference; `beta` is the coefficient in front of the KL term in the advantage (`reward − β · KL`).

## Objective (math)

For a prompt `x` and sampled completion `y` with judge reward `R`. No clipping, no value head. `β` is the KL weight in the advantage (not in a separate regulariser).

```text
A_t            =  R  −  β · KL_t                              (per-token advantage)

ℒ_REINFORCE    =  − ∑_t  A_t · log π_θ(y_t | x, y_<t)
```

## Dataset format

Same as the other online loops: only a `prompt` field at training time. The completion is sampled from the policy and the scalar reward comes from the judge.

The bundled default is `mlx-community/Human-Like-DPO`.

## When to use it

Educational / minimal RLHF. With a modern LLM judge, REINFORCE is competitive with PPO for short completions and is much simpler to debug. It is also the algorithm most sensitive to judge quality and learning rate — start with `lr=5e-6` and `beta=0.05`.

## RLHF-REINFORCE-specific settings

In addition to the [shared SFT substrate](SFT#shared-configuration-reference):

| Setting | Default | What it actually changes |
|---|---|---|
| `beta` | `0.1` | Coefficient on the KL term in the per-token advantage (`A = R − β · KL`). `0` disables KL regularisation. |
| `judge` | `Qwen/Qwen3-0.6B` | LLM judge that produces the scalar reward. Loaded once, called once per (prompt, completion). |
| `max_completion_length` | `128` | Max tokens sampled per completion. REINFORCE is variance-sensitive, so shorter completions usually help. |
| `reference_model_path` | `—` | Frozen reference used to compute the per-token KL. Empty ⇒ second copy of the base model. |

## In the app

On the **Train** tab, RLHF-REINFORCE shows an **Online Preference** block on top of the shared form:

- **Completion** → `max_completion_length` (keep this small — REINFORCE variance scales with trajectory length; `128` is a good default).
- **Judge** picker (segmented LLM / User): LLM → `judge` (the model that produces the scalar reward); User → `judge = "human"`.
- **Judge model** field (shown when Judge = LLM).
- **Dataset Columns** — Prompt.

Not exposed in the UI for this mode (use app defaults; edit via YAML): `beta` (`0.1`, KL weight in the per-token advantage), `judge_system` (the rubric; not surfaced for REINFORCE in the UI — set it in YAML if you want a specific rubric), `reference_model_path` (empty ⇒ second copy of the base model). No temperature field — REINFORCE samples at the policy's default temperature.

Shared form: Model & Data; Fine-tune (LoRA/DoRA/Full + Quantization); Training Settings; Output. (QAT is not applicable to online loops.)

## Tips & gotchas

- Use a small `max_completion_length` (128 is a good default). REINFORCE variance scales with trajectory length.
- Prefer `lr=5e-6` to `1e-5`; REINFORCE has no clipping to catch a too-aggressive step.
- Always log `rewards` and `kl_penalty` — if `kl_penalty` is rising while `rewards` plateaus, the policy is drifting in a way the judge does not see.

## References

- REINFORCE (Williams, 1992) applied to LLM RLHF with an LLM judge and a per-token KL penalty.
- Implementation: `vendor/mlx-lm-lora/mlx_lm_lora/trainer/rlhf_reinforce_trainer.py`.

## See also

- [PPO](PPO) · [GRPO](GRPO) · [Online-DPO](Online-DPO) · [Algorithm Guide](Algorithm-Guide)