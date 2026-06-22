# GRPO

**Family:** Reinforcement / online · **Reference model:** optional · **Judge:** reward functions · **QAT:** no

> RL without a learned reward model: sample a group, score with functions you write, optimise a group-relative advantage.

## Overview

**GRPO (DeepSeekMath, 2024)** is an RL loop that does not need a learned reward model. For every prompt it samples `group_size` completions from the current policy, scores each completion with one or more user-supplied **reward functions** (format checks, accuracy checks, int-format, etc.), and computes a group-relative advantage: `A_i = (r_i − mean(r)) / (std(r) + ε)`. The PPO-style clipped objective is then applied at the token level.

The KL term against the reference model is optional but recommended when the policy starts to drift in a way the rewards do not penalise.

## Intuition

- The reward is whatever functions you ship — string-matching accuracy, integer/format checks, XML-tag counting, etc. The default set is the `r1_*` family (DeepSeek-R1 style format + accuracy rewards).
- Advantages are computed per prompt group, so the absolute scale of the reward functions does not matter — only their *relative* ordering within a group. This is what makes GRPO robust to reward function magnitude.
- Importance sampling lets you decide whether the PPO ratio is a per-token quantity (default, `token`) or averaged across the sequence (`sequence`, more stable per the GSPO paper).
- KL is optional. Set `beta = 0.0` to disable it; the trainer falls back to a Schulman-style unbiased estimator for logging only.

## Objective (math)

For a prompt with `G` sampled completions and reward functions `{r_k}` with weights `{w_k}`. `ε_low` and `ε_high` (`epsilon` and `epsilon_high`) are the asymmetric clip bounds from DAPO. `importance_sampling_level` decides whether `ratio` is per-token or averaged across the sequence.

```text
R_i        =  ∑_k  w_k · r_k(prompt, y_i)                          (total reward)

A_i        =  ( R_i  −  mean_j R_j )  /  ( std_j R_j + 1e−4 )      (group-normalised advantage)

ratio_i,t  =  π_θ(y_i,t)  /  π_ref(y_i,t)                          (importance ratio)

ℒ_clip     =  − min( ratio · A,  clip( ratio, 1−ε_low, 1+ε_high ) · A )

ℒ_KL       =  β · ( ratio · (π_ref / π_θ)  −  log(π_ref / π_θ)  −  1 )    (unbiased KL)

ℒ_GRPO     =  ( ℒ_clip  +  ℒ_KL )   averaged over valid tokens
```

## Dataset format

GRPO needs at minimum a `prompt` field and, for the default `r1_*` reward functions, an `answer` field. The 4-tuple the trainer produces per row is `(prompt_tokens, answer_tokens, prompt_text, answer_text)`; an optional 5th element is the `type` used to switch reward functions per category.

The bundled default is `mlx-community/Dolci-Think-RL-7B-2k`, a reasoning dataset.

## When to use it

When you have a *verifiable* reward (math correctness, code execution, format compliance) rather than a labelled preference dataset. GRPO is the workhorse behind recent reasoning models (DeepSeek-R1, Qwen3-Instruct reasoning mode). Expect completions to look very different from SFT outputs — that is the point.

## GRPO-specific settings

In addition to the [shared SFT substrate](SFT#shared-configuration-reference):

| Setting | Default | What it actually changes |
|---|---|---|
| `group_size` | `4` | Completions sampled per prompt. Higher = lower-variance advantage, more compute per step. |
| `beta` | `0.1` | KL penalty coefficient against the reference. `0` disables KL. |
| `epsilon` / `epsilon_high` | `1e-4 / —` | Asymmetric PPO clip (`ε_low`, `ε_high`). If `epsilon_high` empty, both bounds default to `epsilon`. |
| `max_completion_length` | `512` | Max tokens sampled per completion. Drives time-per-step. |
| `temperature` / `top_p` / `top_k` / `min_p` | `0.8 / 0.95 / 20 / 0.0` | Sampler settings for in-loop generation. `temperature=0` is invalid. |
| `reward_functions` | `—` | Comma-separated reward function names. Empty ⇒ default `r1_*` family (`r1_accuracy`, `r1_int`, `r1_strict_format`, `r1_soft_format`, `r1_count_xml`). |
| `reward_functions_file` | `—` | Path to a Python file that registers functions with `@register_reward_function()`, loaded via `load_reward_functions_from_file` (see [Custom reward functions](#custom-reward-functions)). |
| `reward_weights` | `—` | Comma-separated weights matching the reward function list. Empty = all 1.0. |
| `importance_sampling_level` | `—` | `token` (default), `sequence`, or empty. `sequence` averages the log-ratio per sequence (GSPO). |
| `grpo_loss_type` | `grpo` | `grpo` (mean over all tokens), `bnpo` (normalised by actual token count), `dr_grpo` (divided by `batch_size · max_tokens`). |
| `reference_model_path` | `—` | Reference model used for KL and (when `importance_sampling_level != none`) the importance ratio. |

## In the app

On the **Train** tab, GRPO exposes two algorithm-specific blocks on top of the shared form.

**Preference And Judge** (shared with DPO/CPO/ORPO):
- **Beta** → `beta` (KL weight; set `0` to disable KL).
- **Reference model path** → `reference_model_path` (shown because GRPO uses a reference for KL and the importance ratio; empty ⇒ second copy of the base model).

**GRPO Generation And Rewards**:
- **Group** → `group_size`; **Completion** → `max_completion_length`; **Temp** → `temperature`; **Epsilon** → `epsilon`.
- **Top P** → `top_p`; **Top K** → `top_k`; **Min P** → `min_p`; **Epsilon high** → `epsilon_high`.
- **GRPO Loss** picker: GRPO / BNPO / DR GRPO → `grpo_loss_type` (`grpo` / `bnpo` / `dr_grpo`).
- **Importance** picker: Default / Token / Sequence → `importance_sampling_level` (Default = `token`).
- **Default Reward Functions** — a selectable list of the five built-in `r1_*` functions with a **Use All Defaults** button. Leave the custom list empty to use all backend defaults; selecting rows writes the function names passed to the trainer.
- **Custom reward function names** (comma-separated) → `reward_functions`.
- **Reward weights**, e.g. `[2.0, 0.5, 0.5, 0.5, 0.5]` → `reward_weights`.
- **Reward functions Python file** + **Import** button (file picker for `.py` / `.txt`) → `reward_functions_file`.

Shared form (every algorithm): **Model & Data** (base model, dataset, LM Studio export name); **Fine-tune** (`train_type` LoRA/DoRA/Full, with LoRA Settings — Layers, Rank, Scale, Dropout — for LoRA/DoRA, and Quantization None/4/6/8/MXFP4); **Training Settings** (Iterations, Epochs, Batch, Max Seq, Seed, Learning Rate, Optimizer, LR Schedule, Report/Eval/Save, Val Batches, Gradient accumulation, Sequence step size, Test Batches; grad-checkpoint / mask-prompt / fuse toggles); **Dataset Columns** (Prompt, Answer, Type — the columns the reward functions read); **Output** (run folder name); **QAT** (not applicable to GRPO).

## Custom reward functions

### How it works

GRPO scores each sampled completion with one or more reward functions you supply. The backend keeps a global registry (`REWARD_REGISTRY`) in `mlx_lm_lora/trainer/grpo_reward_functions.py`. Functions register themselves with the `@register_reward_function()` decorator; the trainer resolves the names you list in `reward_functions` via `get_reward_function(name)` and sums their weighted outputs into the per-completion reward `R_i`.

To use your own:

1. Write a `.py` file that imports `register_reward_function` and decorates your functions.
2. Point **Reward functions Python file** at it (or set `reward_functions_file` in YAML). The loader (`load_reward_functions_from_file` in `train.py`) execs the file via `importlib`, so the decorators run and populate the registry at startup.
3. List the registered names in **Custom reward function names** (comma-separated) → `reward_functions`. Leave it empty to use the five built-in `r1_*` defaults.
4. Optionally set **Reward weights** (same length as the function list) → `reward_weights`. Empty ⇒ all weights `1.0`.

### Function signature

Every reward function has the same signature:

```python
RewardFunctions = Callable[[List[str], List[str], List[str], Optional[List[str]]], List[float]]

def my_reward(prompts, completions, answer, types=None) -> list[float]:
    ...
```

- `prompts` — list of prompt strings (one per completion in the group)
- `completions` — list of sampled completion strings
- `answer` — list of reference answer strings from the dataset
- `types` — optional list of per-row category tags (the dataset `type` column → **Type** in Dataset Columns), used to switch reward logic per category
- returns — a list of float rewards, one per completion

The absolute scale does not matter — GRPO normalises rewards within each prompt group into advantages `(r − mean) / (std + ε)`. Only the *relative ordering* within a group matters, which is what makes GRPO robust to reward-function magnitude.

### Built-in defaults

Registered in `grpo_reward_functions.py`, shown in the app's **Default Reward Functions** list:

| App label | Function | Reward |
|---|---|---|
| Accuracy | `r1_accuracy_reward_func` | `2.0` when the extracted `<answer>` exactly matches the dataset answer |
| Integer Answer | `r1_int_reward_func` | `0.5` when the extracted `<answer>` is digit-only |
| Strict Format | `r1_strict_format_reward_func` | `0.5` for strict `<think>…</think><answer>…</answer>` output |
| Soft Format | `r1_soft_format_reward_func` | `0.5` when think/answer tags appear in the right order with content |
| XML Count | `r1_count_xml` | small score for exactly one set of tags, with a trailing-text penalty |

These expect `<answer>…</answer>` (and reasoning/think tags) in the completion — the DeepSeek-R1 style. If your data does not use that structure, write a custom function.

### Example custom reward file

Save as e.g. `my_rewards.py` and point **Reward functions Python file** at it:

```python
import json
from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function


@register_reward_function("exact_match")
def exact_match(prompts, completions, answer, types=None):
    """2.0 when the completion text exactly matches the reference answer."""
    return [2.0 if c.strip() == a.strip() else 0.0
            for c, a in zip(completions, answer)]


@register_reward_function("valid_json")
def valid_json(prompts, completions, answer, types=None):
    """1.0 for a JSON object, 0.5 for any valid JSON, 0.0 otherwise."""
    scores = []
    for c in completions:
        try:
            obj = json.loads(c.strip())
            scores.append(1.0 if isinstance(obj, dict) else 0.5)
        except Exception:
            scores.append(0.0)
    return scores


@register_reward_function("concise")
def concise(prompts, completions, answer, types=None):
    """1.0 for short answers, decaying to 0 as word count passes ~200."""
    return [max(0.0, 1.0 - len(c.split()) / 200.0) for c in completions]


@register_reward_function("keyword_bonus")
def keyword_bonus(prompts, completions, answer, types=None):
    """1.0 when the completion contains 'therefore', else 0.0."""
    return [1.0 if "therefore" in c.lower() else 0.0 for c in completions]
```

Then in the app set **Custom reward function names** to e.g. `exact_match,valid_json,concise,keyword_bonus` and **Reward weights** to e.g. `[2.0, 1.0, 0.5, 0.5]`.

### A category-aware example

Use the `types` argument to switch reward logic per row (the dataset `type` column maps to **Type** in Dataset Columns):

```python
import json
from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function


@register_reward_function("by_type")
def by_type(prompts, completions, answer, types=None):
    scores = []
    types = types or [None] * len(completions)
    for c, a, t in zip(completions, answer, types):
        if t == "math":
            scores.append(2.0 if c.strip() == a.strip() else 0.0)
        elif t == "json":
            try:
                json.loads(c.strip())
                scores.append(1.0)
            except Exception:
                scores.append(0.0)
        else:
            scores.append(0.5 if c.strip() else 0.0)
    return scores
```

### Example YAML

```yaml
train_mode: grpo
model: mlx-community/Qwen3-0.6B-4bit
dataset:
  - mlx-community/Dolci-Think-RL-7B-2k
group_size: 4
max_completion_length: 512
temperature: 0.8
beta: 0.1                  # KL weight (0 disables KL)
epsilon: 1.0e-4
grpo_loss_type: grpo
importance_sampling_level: sequence
reward_functions: exact_match,by_type,concise
reward_weights: [2.0, 1.5, 0.5]
reward_functions_file: ~/my_rewards.py
```

## Tips & gotchas

- Start with the default DeepSeek-R1 reward family. They are designed for `<reasoning>...</reasoning><answer>...</answer>` completions; if your data does not have that structure, write a custom reward function.
- The most common failure mode is `hit_max_tokens_ratio = 1.0` — the model is generating until the limit and never reaching the answer tag. Lower `max_completion_length` or strengthen the format reward.
- `clip_ratio_total` should be in the 0.05–0.2 range. Below that the advantage signal is too weak, above that the policy is moving too aggressively per step.
- `importance_sampling_level = sequence` is a free stability win for reasoning tasks with long completions.
- GRPO is the slowest of the loops because every step does a generation pass; expect 4–8× the wall-clock time of an SFT step at the same `batch_size`.

## References

- DeepSeek-AI, 2024, *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO). DAPO for asymmetric clipping; GSPO for sequence-level importance sampling.
- Implementation: `vendor/mlx-lm-lora/mlx_lm_lora/trainer/grpo_trainer.py`, `grpo_reward_functions.py`.

## See also

- [Online-DPO](Online-DPO) · [XPO](XPO) · [PPO](PPO) · [RLHF-Reinforce](RLHF-Reinforce) · [Algorithm Guide](Algorithm-Guide)