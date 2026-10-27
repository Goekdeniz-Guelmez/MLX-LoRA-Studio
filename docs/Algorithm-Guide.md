# Choose a training method

There are 12 training algorithms in MLX LoRA Studio. They solve different problems, need different data, and have different costs. Start with the evidence you can provide, then use the individual references for objectives, settings, and implementation detail.

## A practical decision path

| What you have | Start here | Why |
|---|---|---|
| Good examples of the behavior you want | [SFT](SFT) | Teaches the model to produce demonstrations you provide. |
| Two answers to the same prompt, with one preferred | [DPO](DPO) | Learns from a fixed set of chosen and rejected answers. |
| Preference pairs, but the reference model does not fit your memory budget | [CPO](CPO) or [ORPO](ORPO) | Avoids a separate frozen reference model during the main preference loss. |
| Preference pairs and a specific reason to constrain hidden-state alignment | [DSLA](DSLA) | Adds a latent alignment term to DPO, CPO, or ORPO. |
| A targeted reasoning branch repair in the Antidoom data format | [FTPO](FTPO) | Focuses on a known final-token decision instead of broad behavior. |
| Prompts and a reliable programmatic reward for generated answers | [GRPO](GRPO) or [KLPO](KLPO) | Samples new responses and trains against reward signals. |
| Prompts, generated answers, and a consistent judge | [Online DPO](Online-DPO) or [XPO](XPO) | Builds preference pairs during training. XPO adds an exploration bonus. |
| A mature rollout and reward setup that needs policy-gradient control | [REINFORCE](RLHF-Reinforce) or [PPO](PPO) | Runs online RLHF; the added control also brings tuning and compute overhead. |

If you are unsure, start with a short SFT run and evaluate real examples. Preference and reinforcement methods are not automatic upgrades; they need stronger data or reward signals than SFT.

## Supervised learning

### SFT

Supervised Fine-Tuning trains on prompt and completion examples, chat messages, or plain text. It is the most useful baseline for teaching a format, response style, or narrow task. Make the training examples resemble real requests, preserve the model's chat template, and keep a clean validation split. Inspect generations as well as loss.

Choose NLL for the normal baseline. Choose Chunked NLL when long sequences create a memory bottleneck. Dynamic Fine-Tuning weights token errors differently; compare it with NLL on the same held-out task before keeping the added objective.

**Best practice:** start with a small, representative dataset and a short run. Confirm the column mapping and prompt masking before increasing sequence length or adapter rank.

[Read the SFT reference](SFT) · [Open a fine-tuning notebook](Notebooks#supervised-fine-tuning)

## Offline preference optimization

### DPO

Direct Preference Optimization uses chosen and rejected answers for the same prompt and a frozen reference model. It is a strong first preference method when the pairs are curated and the base model already knows the task. The reference anchors policy change, but loading it needs additional memory.

**Best practice:** make the two responses answer the same prompt, ensure the chosen response is consistently better under a clear rubric, start conservatively with beta, and watch validation generations for drift.

[Read the DPO reference](DPO) · [Open a preference notebook](Notebooks#preference-optimization)

### CPO

Contrastive Preference Optimization learns from chosen and rejected answers without a separate reference-model path in its main loss. This can fit a smaller memory budget, but the update has less of an anchor to the starting policy.

**Best practice:** use clean, comparable preference pairs and a conservative learning rate. Evaluate for quality regressions and unwanted style changes, not only preference loss.

[Read the CPO reference](CPO)

### ORPO

Odds Ratio Preference Optimization combines a supervised term with a preference term in one stage. It needs chosen and rejected responses but no frozen reference model and no separate SFT warm-up.

**Best practice:** check the expected dataset columns carefully. Run a short baseline first and validate both the preferred behavior and the basic task, since a single objective is doing both jobs.

[Read the ORPO reference](ORPO) · [Open an ORPO notebook](Notebooks#preference-optimization)

### DSLA

Directional and Similarity-aware Latent Alignment adds a hidden-state alignment regularizer to a DPO, CPO, or ORPO preference objective. It is a specialist extension for experiments where latent similarity and direction matter.

**Best practice:** establish a preference-only baseline before adding the latent term. Change one latent setting at a time and watch the latent margin metrics as well as output quality.

[Read the DSLA reference](DSLA)

### FTPO

Final Token Preference Optimization targets a known reasoning branch using the specialized Antidoom data schema. It is a focused repair method, not a general substitute for SFT or DPO.

**Best practice:** verify that each row contains the complete rendered context, rejected continuation, and accepted continuations. Ordinary chosen and rejected preference rows are not interchangeable with FTPO rows.

[Read the FTPO reference](FTPO)

## Online preference and reinforcement learning

### GRPO

Group Relative Policy Optimization samples multiple completions for each prompt, scores them with reward functions, and compares their relative outcomes. It is useful when generated outputs can be scored reliably, including verifiable tasks. Sampling adds substantial work to each step.

**Best practice:** validate the reward on hand-written examples before training. Keep rewards specific, bounded, and difficult to game; inspect sampled outputs often and begin with a small group and short run.

[Read the GRPO reference](GRPO) · [Open a GRPO notebook](Notebooks#reinforcement-learning)

### KLPO

KL-Regularized Policy Optimization samples one completion per prompt and uses the GRPO reward functions to form a reward-tilted target while estimating divergence from the sampling policy. It is critic-free and uses fewer generations per step than GRPO, but reward quality still determines the signal.

**Best practice:** begin with the token route and Monte Carlo estimator described in the reference. Check that terminal rewards are not almost always zero and inspect reward and KL metrics together.

[Read the KLPO reference](KLPO)

### Online DPO

Online DPO samples two completions during training and asks a human or language-model judge to select one. It can adapt the preference data to the current policy, but each label depends on the judge's consistency and rubric.

**Best practice:** write a short, concrete judge rubric and test it on examples where the expected preference is known. Keep a held-out evaluation set that is not judged by the same live training process.

[Read the Online DPO reference](Online-DPO) · [Open an online preference notebook](Notebooks#preference-optimization)

### XPO

Exploratory Preference Optimization follows the online preference loop and adds a bonus for moving away from the reference. It can help when online DPO repeatedly produces near-identical choices, but an overly strong bonus can encourage unhelpful drift.

**Best practice:** establish a stable Online DPO run first. Increase exploration only when measured generations show the policy is stuck, then reduce the bonus as the run progresses.

[Read the XPO reference](XPO)

### RLHF Reinforce

REINFORCE updates the policy using sampled completions and judge or reward feedback. It is a direct policy-gradient approach with fewer moving parts than PPO, but its updates can have high variance.

**Best practice:** begin with a small learning rate, a stable reward scale, and short runs. Track reward distributions and evaluate manually; a rising reward is not enough if the reward can be exploited.

[Read the REINFORCE reference](RLHF-Reinforce)

### PPO

Proximal Policy Optimization uses a clipped policy update and reference or judge feedback to limit how far each step moves. It offers more explicit control over an online RL loop and is more sensitive to rollout, reward, and optimization settings.

**Best practice:** first confirm the reward and generation pipeline on a small sample. Keep the clip range and learning rate conservative, watch KL and reward together, and stop when outputs regress even if the scalar reward rises.

[Read the PPO reference](PPO)

## The method around the method

The training objective is only one part of a run. The adapter method, precision, optimizer, sequence length, and data quality all change memory use and behavior.

- [LoRA](LoRA) trains small low-rank adapter matrices and is a sensible first choice.
- [DoRA](DoRA) separates weight magnitude and direction while keeping an adapter-style workflow.
- [QLoRA](QLoRA) loads a quantized base to reduce memory use; available precision and compatibility depend on the model and trainer.
- [QAT](QAT) simulates quantization during training for supported workflows. It is useful when a quantized deployment is the target.
- [Full fine-tuning](Full-Fine-Tuning) updates all trainable model weights and needs substantially more memory.
- [Optimizers](Optimizers) explains the available update rules and how to begin.

For a resource-conscious first experiment, use a compatible instruct model, LoRA, a short sequence length, gradient checkpointing, and a small dataset. Increase one source of capacity at a time. See [live metrics](Live-Metrics) and the [training workflow](Train) for the app controls.

## Try the examples

The notebook collection has runnable supervised, preference, and reinforcement-learning walkthroughs. They are useful for inspecting the data shapes and configuration before you build a larger experiment.

[Browse example notebooks](Notebooks) · [Open the notebook repository](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks)
