# Runs and checkpoints

The Runs view turns an experiment into a record you can revisit. Studio reads runs from its output folder and shows the configuration, training log, status, metrics, and saved adapter.

<div class="doc-screenshot">
  <img src="https://raw.githubusercontent.com/Goekdeniz-Guelmez/MLX-LoRA-Studio/main/Sources/Media/app_runs.png" alt="MLX LoRA Studio Runs view showing active and previous jobs with status and training charts" loading="lazy">
</div>

## Keep a useful record

Each run stores its configuration and logs next to the resulting adapter and checkpoints. The app tracks active and completed jobs, reports available metrics, and can reveal the run folder in Finder.

## Continue from an adapter

Use a compatible saved adapter as the starting point for a new run. This supports staged workflows such as SFT followed by DPO, or continuing a run after inspecting its output. Keep the base model and adapter configuration aligned when resuming.

## Compare the result

Open a run to inspect its settings, final metrics, and log. Compare like with like: model, dataset, token length, and training method all influence memory and speed. Use a separate held-out evaluation set to compare model behavior.

The run archive is stored locally in the user's Application Support folder. Uploading an adapter is a separate action in [Upload to Hugging Face](Upload-to-HF).

Older synthetic-data run folders remain readable in the archive. Creating synthetic datasets is not part of the current app workflow.
