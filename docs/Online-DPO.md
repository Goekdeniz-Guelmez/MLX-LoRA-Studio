# Online DPO

**Family:** Reinforcement / online · **Reference model:** yes · **Judge:** LLM or human · **QAT:** no

> DPO where the chosen and rejected completions are sampled at training time and labelled by a judge — no stale off-policy dataset.

## Overview

**Online DPO** is DPO where the `chosen` and `rejected` completions are sampled at training time instead of read from a static dataset. For every prompt the trainer draws two completions, asks a **judge** (an LLM or a human) which one is better, treats that as the preference pair, and runs the DPO loss on the fly.

The judge can be `human` (you label pairs interactively), or a Hugging Face model identifier / local path that the runner loads and prompts with a pairwise system prompt.

## Intuition

- Every step is: prompt → two completions → judge picks a winner → DPO loss.
- Because the completions come from the *current* policy, the data distribution shifts as training progresses, which avoids the "off-policy stale data" failure mode of vanilla DPO.
- The judge system prompt is a high-leverage string: it controls the rubric. A vague "which is better?" prompt produces noisy labels; a specific "reward conciseness, factual accuracy, and refusal of unsafe content" prompt is much sharper.
- `loss_type` and `delta` behave exactly as in DPO.

## Objective (math)

For each prompt `x`, sample two completions `(y_1, y_2)`, judge picks the winner `w ∈ {0, 1}`. The temperature and judge system prompt together control how informative the labels are; `beta` and `loss_type` are identical to DPO.

```text
y_chosen   =  y_w
y_rejected =  y_{1−w}

ℒ  =  DPO loss on ( y_chosen, y_rejected )    — see DPO math
```

## Dataset format

Online loops only need a `prompt` field at training time. The completions are sampled from the policy itself, and the (chosen, rejected) pair comes from the judge, not from the dataset.

The bundled default is `mlx-community/Human-Like-DPO` for its small size and standard prompt column.

## When to use it

When you have a strong LLM judge (or a human in the loop) and a base prompt distribution you can keep sampling from. Online DPO avoids the off-policy gap of static DPO and works well with iterative refinement of the same model.

## Online-DPO-specific settings

In addition to the [shared SFT substrate](SFT#shared-configuration-reference):

| Setting | Default | What it actually changes |
|---|---|---|
| `beta` | `0.1` | DPO temperature. |
| `dpo_cpo_loss_type` | `sigmoid` | Loss variant. Same four options as DPO. |
| `delta` | `50.0` | Drift-penalty coefficient for `dpop` loss. |
| `judge` | `Qwen/Qwen3-0.6B` | HF id / local path of the judge LLM, or the literal string `human`. With `human` the runner pauses and asks you to label each pair. |
| `judge_system` | `—` | System prompt sent to the LLM judge. Treat this as the rubric — short, specific, concrete. |
| `max_completion_length` | `512` | Max tokens sampled per completion in the in-loop generation. |
| `temperature` | `0.8` | Sampling temperature for the policy. Lower ⇒ both completions look more similar ⇒ harder comparisons for the judge. |
| `reference_model_path` | `—` | Path/HF id of the frozen reference. Empty ⇒ second copy of the base model. |

## In the app

On the **Train** tab, Online DPO shows an **Online Preference** block on top of the shared form:

- **Completion** → `max_completion_length`; **Temp** → `temperature` (sampling temperature for the two policy completions).
- **Judge** picker (segmented LLM / User): LLM → `judge` (a model id / local path); User → `judge = "human"` (the runner pauses for you to label each pair).
- **Judge model** field (shown when Judge = LLM).
- **Judge system prompt** → `judge_system` (shown for the LLM judge — this is the rubric; write it as a concrete, specific criteria list).
- **Dataset Columns** — Prompt (the only column online loops read at training time).

Not exposed in the UI for online modes (they use app defaults; edit via YAML if needed): `beta` (default `0.1`), `dpo_cpo_loss_type` (default `sigmoid`), `delta` (default `50.0`), `reference_model_path` (default empty ⇒ second copy of the base model).

Shared form: Model & Data; Fine-tune (LoRA/DoRA/Full + Quantization); Training Settings; Output. (QAT is not applicable to online loops.)

## Tips & gotchas

- The judge system prompt is the single highest-leverage string in the whole config. Write it as a concrete rubric with bullet points, not as a vague "which is better?".
- Use `temperature ≈ 0.8` and `top_p=0.95` for the policy — too low and both completions are identical, too high and the judge labels look random.
- If you set `judge = "human"`, the runner will pause and ask you to label every pair; budget your time accordingly (or drop `batch_size` to 1).

## References

- Online DPO inherits the DPO objective (Rafailov et al., 2023) applied to on-policy sampled pairs.
- Implementation: `vendor/mlx-lm-lora/mlx_lm_lora/trainer/online_dpo_trainer.py`, `judge.py`.

## See also

- [DPO](DPO) · [XPO](XPO) · [PPO](PPO) · [Algorithm Guide](Algorithm-Guide)