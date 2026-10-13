# KLPO

KL-Regularized Policy Optimization (KLPO) was added in `mlx-lm-lora` 5.8.6. It is a critic-free RL loop for verifiable rewards: it samples **one** complete response per prompt, scores it with the GRPO reward functions, and regresses the policy toward a reward-tilted target while estimating the KL to the sampling policy.

KLPO has no group normalization, no PPO ratio clipping, and no reference model, so it needs fewer generations per step than GRPO.

## Dataset format

The same rows as GRPO: a `prompt` field and, for the default `r1_*` rewards, an `answer` field (`system` and `type` are optional). The bundled default is `mlx-community/Dolci-Think-RL-7B-2k`.

## Controls

| Studio control | Run-spec key | Default | Meaning |
|---|---|---:|---|
| Beta | `beta` | 0.1 | KL regularization strength. |
| Route | `klpo_route` | `token` | `token` regresses every sampled token; `sequence` regresses the whole response. |
| KL estimator | `klpo_kl_estimator` | `mc` | `mc`, `topk`, `binary`, or `full`. |
| MC samples | `klpo_mc_samples` | 128 | Auxiliary draws per prefix for the MC estimator. |
| Top-K head / Tail floor | `klpo_top_k`, `klpo_tail_floor` | 128 / 1e-6 | Settings for the Top-K estimator. |
| Completion / Temp | `max_completion_length`, `temperature` | 512 / 1.0 | Sampler settings. |
| Reward functions, weights, file | `reward_functions`, `reward_weights`, `reward_functions_file` | defaults | Shared with GRPO. |

## Tips

- Start with the token route, the MC estimator, and `beta = 0.1`.
- Lower MC samples or switch to the Binary estimator when memory is tight; Full keeps the whole vocabulary per token.
- Rewards are terminal. A reward that is almost always zero gives KLPO nothing to learn from.

- Implementation: `vendor/mlx-lm-lora/mlx_lm_lora/trainer/klpo_trainer.py`.
