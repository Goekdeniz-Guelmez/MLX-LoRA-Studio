# Settings and onboarding

MLX LoRA Studio is designed to take a Mac from first launch to a reproducible training run without requiring a hand-built Python environment.

## First launch

The onboarding flow checks the system and helps locate or provision a Python environment for the trainer. The release app bundles its helper resources; the backend may install or update packages the first time you use a training feature.

The current bundled mlx-lm-lora backend requires Python 3.11 or later. For the latest version and model compatibility, use the [project releases](https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio/releases) and [trainer documentation](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora).

## Before a run

- Use macOS 14 or later on Apple Silicon. MLX is not supported on Intel Macs.
- Allow time and disk space for model and dataset downloads when they are not already cached.
- Keep enough free unified memory for macOS as well as training. A configuration estimate cannot account for every other process on the machine.
- Check the selected model's license and the dataset's intended use.
- Start with a short run and inspect generated examples before increasing the budget.

## Environment and credentials

Settings lets you inspect or select the Python environment used by the trainer and configure optional Hugging Face access for private assets or uploads. Hugging Face transfers use the network. Training outputs and the run archive remain on the Mac until you choose to publish them.

The [MCP server](MCP-Server) uses its own Python environment and settings so that agent integrations can be managed separately from the training runner.

[Start the training workflow](Train) · [Review local metrics](Live-Metrics)
