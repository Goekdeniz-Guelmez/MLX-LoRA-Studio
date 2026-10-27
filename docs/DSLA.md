# DSLA

Directional and Similarity-aware Latent Alignment (DSLA) was added in `mlx-lm-lora` 5.8.6. It keeps a normal preference objective (DPO, ORPO, or CPO) and adds a regularizer on the model's hidden states, so the chosen answer sits closer to the prompt in latent space and points along the batch's shared "better" direction.

## Dataset format

DSLA reads the same rows as DPO:

```json
{"system": "optional", "prompt": "...", "chosen": "...", "rejected": "..."}
```

The rendered generation prompt must be an exact token prefix of both responses. The bundled default is `mlx-community/Human-Like-DPO`.

## Controls

| Studio control | Run-spec key | Default | Meaning |
|---|---|---:|---|
| Preference objective | `dsla_loss` | `dpo` | `dpo` loads a frozen reference model; `orpo` and `cpo` are reference-free. |
| Beta / Delta / Loss | `beta`, `delta`, `dpo_cpo_loss_type` | 0.1 / 50.0 / sigmoid | Passed to the preference objective as for DPO and CPO. ORPO uses `beta` as its preference-term weight. |
| Latent weight | `latent_weight` | 0.1 | Weight of the hidden-state term. |
| Margin | `latent_margin` | 0.05 | Target prompt-similarity gap between chosen and rejected. |
| Gamma | `latent_gamma` | 10.0 | Sharpness of the soft margins. Must be positive. |
| Variant | `latent_variant` | `both` | `similarity`, `direction`, or their average. |
| Pooling | `latent_pooling` | `answer_mean` | `answer_mean`, `last_token`, `last_k_mean` (last 8 tokens), or `prompt_answer_mean`. |
| Latent layer | `latent_layer` | `final` | `final`, `middle`, `late`, or a zero-based block index. |

## Notes

- Efficient long context is not supported for DSLA. Studio turns it off for DSLA runs; use gradient checkpointing and **Recurrence chunk size** instead.
- QAT works with DSLA. Bits, group size, and start step apply; mode and interval have no effect.
- Live metrics include `latent_sim_margin` and `latent_dir_margin`. Both should rise during training.

## References

This is an implementation-specific method; no standalone paper is cited for this trainer. See the [DSLA trainer source](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora/blob/main/mlx_lm_lora/trainer/dsla_trainer.py).
