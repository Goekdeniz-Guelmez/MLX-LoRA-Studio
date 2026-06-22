# Optimizers

**Foundation** · `optimizer: adam` / `adamw` / `muon`

> How gradients become weight updates. AdamW is the safe default; Muon is a free-lunch speedup on LoRA/DoRA hidden weights of 1B+ models.

## Overview

The built-in guide documents three optimizers — **Adam**, **AdamW**, and **Muon** — but the app's Optimizer picker exposes the full `mlx.optimizers` set (10 in total: SGD, RMSprop, Adagrad, AdaDelta, Adam, AdamW, Adamax, Lion, Adafactor, Muon). The first two documented ones are the workhorses; Muon is a relatively recent addition (Jordan et al., 2024) that has been shown to converge faster on transformer hidden weights. The selection is exposed through `config.optimizer` and the trainer picks the matching class in `mlx.optimizers` at construction time.

## Intuition

- **Adam** keeps a per-parameter exponential moving average of the first moment `m` and the second moment `v` of the gradient, and applies a bias-corrected update. The default `betas = (0.9, 0.999)` work well across a wide range of LRs.
- **AdamW** is Adam with **decoupled weight decay**: the regularisation is `λ · W` added directly to the update rather than appearing inside the gradient. This is the correct way to do L2 regularisation on adaptive optimizers and is the standard choice for transformer fine-tuning.
- **Muon** is "Adam with the second moment replaced by a Newton-Schulz orthogonalisation of the momentum". Concretely, the per-tensor update is projected to (approximately) an orthogonal matrix before being scaled. Empirically faster convergence on hidden 2D weights (QKV and MLP projections) and much worse on 1D weights (biases, norms) — which is why Muon implementations typically pair it with AdamW for the non-hidden params.
- All three are invoked with the same call signature `opt = OptClass(learning_rate=lr, **optimizer_config[opt_name])`, so per-optimizer hyperparameters (`betas`, `eps`, `weight_decay`, `momentum`) are forwarded verbatim from the `optimizer_config` dict.

## Objective (math)

Let `g_t = ∇ℒ(θ_{t−1})` be the gradient at step `t`, and `lr` the learning rate. All three store state in `optimizer.state`; the trainer seeds it from `mx.random.state` for determinism.

**Adam** (Kingma & Ba, 2014):

```text
m_t   =  β₁ · m_{t−1}  +  ( 1 − β₁ ) · g_t               (first moment)
v_t   =  β₂ · v_{t−1}  +  ( 1 − β₂ ) · g_t²              (second moment)
m̂_t  =  m_t / ( 1 − β₁^t )                              (bias correction)
v̂_t  =  v_t / ( 1 − β₂^t )                              (bias correction)
θ_t   =  θ_{t−1}  −  lr · m̂_t / ( √v̂_t + ε )
```

Default in `mlx.optimizers`: `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e−8`. L2 regularisation is *not* applied — use AdamW if you want weight decay.

**AdamW** (Loshchilov & Hutter, 2019):

```text
( m_t, v_t, m̂_t, v̂_t )  ←  Adam update as above
θ_t   =  θ_{t−1}  −  lr · ( m̂_t / ( √v̂_t + ε )  +  λ · θ_{t−1} )
```

The `λ · θ` term is the decoupled weight decay. Default `weight_decay = 0.01`; tune to control how aggressively the model is pulled toward zero (and how much the LoRA/DoRA adapters are encouraged to stay small).

**Muon** (Jordan et al., 2024):

```text
m_t   =  μ · m_{t−1}  +  g_t                       (momentum buffer, μ ≈ 0.95)
O_t   =  NewtonSchulz5( m_t )                      (≈ orthogonalise the momentum)
scale =  √( out · in )                             (spectral-norm-preserving scale)
θ_t   =  θ_{t−1}  −  lr · scale · O_t
```

The Newton–Schulz iteration is a small fixed-point loop (5 steps in the paper) that maps a matrix to its nearest semi-orthogonal one. The update is a single matrix multiply per parameter, so wall-clock cost is comparable to AdamW despite the extra iteration.

## What the settings change

| Setting | Default | What it actually changes |
|---|---|---|
| `optimizer` | `adamw` | Pick `adam`, `adamw`, or `muon`. Class loaded from `mlx.optimizers`, constructed with `learning_rate=lr` plus the matching `optimizer_config` dict. |
| `learning_rate` | `1e-5` | Peak LR. LoRA/DoRA: 1e-5 to 5e-5 (AdamW), 5e-4 to 5e-3 (Muon). Full fine-tuning: 1e-6 to 5e-6 (AdamW). |
| `lr_schedule` | `—` | Optional schedule from `mlx_lm.tuner.utils.build_schedule`. If non-empty, wraps `learning_rate`; otherwise constant. |
| `optimizer_config.adam` | `{}` | Extra kwargs for `optim.Adam`: `betas`, `eps`. |
| `optimizer_config.adamw` | `{}` | Extra kwargs for `optim.AdamW`: `betas`, `eps`, `weight_decay`. Default `weight_decay=0.01`; raise to 0.05 if LoRA magnitudes drift up. |
| `optimizer_config.muon` | `{}` | Extra kwargs for `optim.Muon`: `momentum`, `nesterov`, `weight_decay`. Defaults `momentum=0.95`, `nesterov=True`. |

## When to use which

- **AdamW** is the safe default for everything in this app. Pick it when you do not have a strong reason to use something else.
- **Adam** (without weight decay) is reasonable when you explicitly want to disable regularisation — e.g. the base model is already regularised and you do not want the LoRA adapters penalised.
- **Muon** is a free lunch on the LoRA/DoRA hidden weights of any 1B+ model; it converges in roughly half the iterations to the same loss. Pair it with a higher learning rate (5e-3 to 1e-2) and keep `weight_decay=0.0` unless you specifically want the LoRA magnitudes regularised.

## In the app

On the **Train** tab → **Training Settings**:

- **Optimizer** picker — exposes all 10 `mlx.optimizers` classes: SGD, RMSprop, Adagrad, AdaDelta, Adam, AdamW, Adamax, Lion, Adafactor, Muon → `optimizer`.
- **Learning Rate** → `learning_rate`.
- **LR Schedule** picker (e.g. warmup / cosine) → `lr_schedule`, with **Warmup**, **Decay Fraction**, and **Final LR** fields when a schedule is selected.

Per-optimizer kwargs (`betas`, `eps`, `weight_decay`, `momentum`, `nesterov`) are forwarded from the `optimizer_config` dict — they are not exposed as dedicated fields in the UI. Edit them directly in the run's YAML (`optimizer_config.adamw: {weight_decay: 0.01, betas: [0.9, 0.999]}`) if you need to tune them.

## Tips & gotchas

- If you switch from AdamW to Muon, raise the learning rate by ~100×. The Muon update is parameterised differently and the AdamW default of 1e-5 will under-train.
- Watch the live-metrics loss curve for the first 50 steps after switching optimizers — a sudden divergence almost always means the LR is wrong, not the optimizer choice.
- If you set `weight_decay` for AdamW on a `full` fine-tuning run, expect a small loss bump in the first 100 steps as the regulariser pulls weights toward zero. Normal; usually recovers within an epoch.
- `lr_schedule` is a small DSL from upstream mlx-examples. Common choices: `cosine:<iters>:<min_lr>` for cosine decay, `warmup_cosine:<warmup>:<iters>:<min_lr>` for warm-up followed by decay.

## References

- Kingma & Ba, 2014, *Adam: A Method for Stochastic Optimization*.
- Loshchilov & Hutter, 2019, *Decoupled Weight Decay Regularization* (AdamW).
- Jordan et al., 2024, *Muon* (Newton-Schulz orthogonalised momentum).

## See also

- [LoRA](LoRA) · [DoRA](DoRA) · [Full-Fine-Tuning](Full-Fine-Tuning) · [QLoRA](QLoRA) · [SFT](SFT)