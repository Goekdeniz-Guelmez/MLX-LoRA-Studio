# FTPO

Final Token Preference Optimization (FTPO) is a focused repair objective introduced in `mlx-lm-lora` 3.0.0. It is designed for datasets produced by Liquid AI's Antidoom workflow and updates the next-token distribution at a known reasoning branch.

## Dataset format

Use either a Hugging Face dataset repository ID or a local dataset directory containing `train.jsonl` and, optionally, `valid.jsonl` and `test.jsonl`. In Studio, enter the repository ID in **Model And Data**, choose it from the cached-dataset list, or select **Add HF repo…**. Every row must contain:

```json
{
  "context_with_chat_template": "...full context...",
  "rejected_decoded": "bad next token or continuation",
  "multi_chosen_decoded": ["acceptable continuation A", "acceptable continuation B"]
}
```

Ordinary DPO `prompt`/`chosen`/`rejected` rows are not interchangeable with FTPO rows.

## Controls

| Studio control | Run-spec key | Default | Meaning |
|---|---|---:|---|
| Target MSE λ | `lambda_mse_target` | 0.05 | Penalizes excessive drift on chosen and rejected target logits. |
| Target MSE τ | `tau_mse_target` | 1.0 | Allows this much absolute target-logit drift before its MSE penalty activates. |
| Non-target MSE λ | `lambda_mse` | 0.4 | Keeps unrelated vocabulary logits close to the frozen reference. |
| Logit clip ε | `clip_epsilon_logits` | 2.0 | Sets the chosen/rejected margin and clipping threshold; it must be positive. |

FTPO uses a frozen copy of the selected model as its reference by default. Set **Reference model path** only when a different checkpoint should anchor the repair.

Begin with the defaults, a small learning rate, and a short validation run. FTPO is intentionally narrow: use SFT or DPO for broad behavior changes.
