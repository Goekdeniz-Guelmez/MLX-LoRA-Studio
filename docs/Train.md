# Train

The Train screen is where model, data, objective, adapter, and resource choices come together. Start with a model already in your Hugging Face cache or choose one to download, select a dataset, and pick the training method that matches its structure.

## Shape the run

- **Model and data:** choose a base model and dataset, then confirm the dataset columns and chat template.
- **Training objective:** select among the supported supervised, preference, and reinforcement methods in the [algorithm guide](Algorithm-Guide).
- **Adaptation:** use [LoRA](LoRA) or [DoRA](DoRA) to train an adapter, [QLoRA](QLoRA) to load a low-bit base model, or [full fine-tuning](Full-Fine-Tuning) when you intend to update all weights.
- **Optimization:** set learning rate, schedule, batch size, gradient accumulation, sequence length, evaluation, and checkpoint intervals.
- **Output:** choose a run name and output location; resume from a saved adapter when you want to continue a compatible run.

The form exposes settings that are relevant to the selected method. Its YAML configuration can be inspected and re-run with the same command-line trainer.

## Check the resource estimate

Before starting, Studio estimates memory for the selected model and settings and monitors live wired and active memory. ResourceGuard checks system memory pressure and can block a run that is unlikely to fit. The estimate is a planning aid, not a guarantee: model architecture, sequence length, quantization, batch size, and concurrent Mac workloads affect actual use.

If a run is too large, try a smaller model, a shorter sequence, a smaller micro-batch, gradient checkpointing, or a lower-memory adaptation method. Increase one setting at a time so you can see which change matters.

## Start small

For a first experiment, use a compatible instruct model, LoRA, a short sequence length, and a small representative dataset. Run a few steps, inspect the output and logs, and check that loss and validation behave as expected before committing to a longer run.

## Control a run

Start, pause, resume, or stop the active job from the toolbar. The sidebar shows progress and live memory. When training ends, the configuration, log, and adapter are available in [Runs](Runs).

See the [live metrics guide](Live-Metrics), [setup notes](Settings-and-Onboarding), and [example notebooks](Notebooks).
