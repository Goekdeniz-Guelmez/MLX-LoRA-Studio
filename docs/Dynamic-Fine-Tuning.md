# Dynamic fine-tuning

Selectable SFT loss functions were added in `mlx-lm-lora` 3.0.0 and remain available in the current 5.10.2 trainer. Choose one in **Train → Training Settings → SFT loss**.

| Loss | Run-spec value | Use it when |
|---|---|---|
| NLL | `nll` | You want standard next-token cross-entropy. |
| Chunked NLL | `chunked_nll` | Vocabulary logits are the memory bottleneck and you want bounded peak loss memory. |
| Dynamic fine-tuning | `dft` | You want training to emphasize difficult, low-confidence tokens instead of spending equal weight on tokens the model already predicts confidently. |

Dynamic fine-tuning changes only the SFT objective. Dataset formats, LoRA/DoRA/full adaptation, quantized loading, checkpointing, reporting, and export work the same way as ordinary SFT.

Start with the same learning rate you would use for NLL and compare validation loss and downstream generations. The numerical loss scale is not necessarily directly comparable across objectives, so use held-out behavior—not only the plotted scalar—to choose a loss.

Run-spec example:

```json
{
  "train_mode": "sft",
  "sft_loss_type": "dft"
}
```

## References

- [Wu et al. (2025), *On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification*](https://arxiv.org/abs/2508.05629).
- Implementation: `vendor/mlx-lm-lora/mlx_lm_lora/trainer/sft_trainer.py`.
