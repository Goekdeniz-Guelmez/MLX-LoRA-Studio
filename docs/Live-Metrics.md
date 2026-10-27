# Live metrics

Training should be inspectable while it runs. The Live Metrics view reads the trainer's reports and makes the most useful signals visible alongside the run log.

<div class="doc-screenshot">
  <img src="https://raw.githubusercontent.com/Goekdeniz-Guelmez/MLX-LoRA-Studio/main/Sources/Media/app_live_metrics.png" alt="MLX LoRA Studio live metrics view with loss, throughput, memory, learning rate, and recent training reports" loading="lazy">
</div>

## What you can follow

- **Loss and validation loss** show how the reported objective changes over time. A lower training loss alone does not prove that the model is more useful.
- **Learning rate and gradient norm** help explain the size and stability of updates.
- **Iterations per second and tokens per second** describe throughput for the current run. Compare runs only when their models, sequence lengths, batches, and hardware are comparable.
- **Memory** shows the trainer's reported usage; the sidebar also monitors live system memory and the estimate for the selected configuration.
- **Progress and logs** give run status and the trainer output behind each metric.

The trainer decides which values it reports. The page surfaces reported metrics and recent steps; it does not invent missing values.

## Read the curves with the task

Loss is a training signal, not a quality score. Compare validation behavior, then test the saved adapter on prompts that resemble the intended use. For preference and reinforcement runs, check reward, KL, and algorithm-specific metrics as well as actual generations.

If a curve changes abruptly, inspect the corresponding log lines, dataset batch, and learning rate before changing several parameters. The [Runs archive](Runs) preserves reports and configuration for later comparison.
