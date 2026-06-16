from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

README_AUTHOR = "MLX-LoRA-Studio"
README_STAMP = "Created with MLX LoRA Studio"
README_TAGLINE = f"Created with {README_AUTHOR} · {README_STAMP}"

SPARK_GLYPHS = "▁▂▃▄▅▆▇█"
SPARK_WIDTH = 40

LOSS_CURVE_IMAGE_NAME = "loss_curve.png"


def studio_log(message: str) -> None:
    print(f"[Studio] {message}", flush=True)


def require_hub() -> tuple[Any, Any]:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for uploads. Install it with `pip install huggingface_hub`."
        ) from exc
    return HfApi, create_repo


def clean_string(value: Any) -> str:
    return str(value or "").strip()


def validate_repo(repo: str, label: str) -> str:
    if not repo or "/" not in repo:
        raise ValueError(
            f"{label} must be a Hugging Face repo id like `username/name`."
        )
    return repo


def validate_path(path: str, label: str) -> Path:
    local = Path(path).expanduser()
    if not local.exists():
        raise FileNotFoundError(f"{label} does not exist: {local}")
    if not local.is_dir():
        raise ValueError(f"{label} must be a folder: {local}")
    return local


# MARK: - Run-folder resolution
#
# The HF upload picks `<runFolder>/adapters` (or a custom folder the
# user typed). We want to attach the run's training settings + metrics
# to the README, so the first job is figuring out which parent folder
# actually contains the Studio metadata. The two known anchor files
# are `run_spec.json` (the TrainingConfig payload) and `metrics.json`
# (the loss-curve history written by the runner after this change).
#
# We try the model folder's parent first, then the parent of that,
# then the model folder itself. The first candidate that contains
# either anchor file wins. If none of them do, we treat the upload as
# a "plain" folder and fall back to the minimal README.


def resolve_run_folder(model_path: Path) -> Path | None:
    candidates: list[Path] = []
    parent = model_path.parent
    candidates.append(parent)
    if parent.parent != parent:
        candidates.append(parent.parent)
    candidates.append(model_path)
    for candidate in candidates:
        if (candidate / "run_spec.json").exists() or (
            candidate / "metrics.json"
        ).exists():
            return candidate
    return None


def load_spec(run_folder: Path) -> dict[str, Any] | None:
    spec_path = run_folder / "run_spec.json"
    if not spec_path.exists():
        return None
    try:
        with spec_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def load_metrics(run_folder: Path) -> list[dict[str, Any]]:
    metrics_path = run_folder / "metrics.json"
    if not metrics_path.exists():
        return []
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def resolve_synthetic_run_folder(dataset_path: Path) -> Path | None:
    candidates = [dataset_path, dataset_path.parent]
    if dataset_path.parent.parent != dataset_path.parent:
        candidates.append(dataset_path.parent.parent)
    for candidate in candidates:
        if (candidate / "synthetic_spec.json").exists():
            return candidate
    return None


def load_synthetic_spec(run_folder: Path) -> dict[str, Any] | None:
    spec_path = run_folder / "synthetic_spec.json"
    if not spec_path.exists():
        return None
    try:
        with spec_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


# MARK: - Sparkline rendering
#
# Two backends, picked at runtime. We try matplotlib first because it
# produces a clean line chart that shows up in the HF preview; if the
# import fails (matplotlib isn't always installed) we fall back to a
# unicode block sparkline that needs zero dependencies and still gives
# a visual hint of the curve shape.


def render_loss_png(
    train_values: list[float],
    val_values: list[float],
    out_path: Path,
    title: str = "Training Loss",
) -> bool:
    """Render a PNG of the loss curve with matplotlib. Returns True on
    success, False if matplotlib isn't importable or the render
    failed for any reason — the caller falls back to the unicode
    sparkline in that case."""
    if not train_values and not val_values:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless; no display required
        import matplotlib.pyplot as plt
    except Exception:
        return False
    try:
        fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=140)
        if train_values:
            ax.plot(train_values, label="Train loss", color="#ff8c42", linewidth=1.8)
        if val_values:
            ax.plot(
                val_values,
                label="Validation loss",
                color="#4a90e2",
                linewidth=1.8,
                linestyle="--",
            )
        ax.set_title(title, fontsize=11, color="#222")
        ax.set_xlabel("Report step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_path, format="png")
        plt.close(fig)
        return True
    except Exception:
        return False


def render_loss_sparkline(values: list[float], width: int = SPARK_WIDTH) -> str:
    """Collapse `values` into `width` buckets and render as a string of
    block characters. Returns an empty string for empty / singleton
    inputs so the caller can omit the line entirely."""
    if len(values) < 2:
        return ""
    # Downsample by averaging `width` evenly-spaced buckets. For very
    # long runs this gives a representative shape; for short runs (a
    # handful of points) we still get one bar per report.
    if len(values) <= width:
        buckets = [float(v) for v in values]
    else:
        bucket_size = len(values) / width
        buckets = []
        for i in range(width):
            start = int(i * bucket_size)
            end = int((i + 1) * bucket_size)
            if end <= start:
                end = start + 1
            buckets.append(sum(values[start:end]) / max(end - start, 1))
    lo, hi = min(buckets), max(buckets)
    span = max(hi - lo, 1e-12)
    glyphs = []
    for v in buckets:
        idx = int(round((v - lo) / span * (len(SPARK_GLYPHS) - 1)))
        glyphs.append(SPARK_GLYPHS[idx])
    return "".join(glyphs)


# MARK: - Settings block
#
# The HF README shows the *full* TrainingConfig so anyone who lands on
# the repo can reproduce the run. We group fields into headed
# sections, mirroring the Runs page detail sheet, and skip empty
# optional fields so the block doesn't drown in `- foo: `.


# Map of (group title, list of (key, label, formatter)) tuples. Each
# formatter turns the JSON value into a printable string; the default
# `str(v)` covers strings, ints, and bools.
def _f_str(v: Any) -> str:
    return clean_string(v) or "—"


def _f_int(v: Any) -> str:
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return "—"


def _f_float(v: Any) -> str:
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return "—"


def _f_bool(v: Any) -> str:
    return "✅" if v else "❌"


def _f_optional_str(v: Any) -> str:
    s = clean_string(v)
    return s if s else "—"


def _settings_rows(spec: dict[str, Any]) -> list[tuple[str, list[tuple[str, str]]]]:
    """Returns the grouped settings rows. Each group is (title, [(label, value), ...]).
    Empty / null values are dropped from the row list so the README
    doesn't list `lr_schedule: ` for runs that used a constant LR."""
    model_short = clean_string(spec.get("model"))
    model_label = f"`{model_short}`" if model_short else "—"

    lora = (
        spec.get("lora_parameters")
        if isinstance(spec.get("lora_parameters"), dict)
        else {}
    )
    lora_rank = _f_int(lora.get("rank", spec.get("rank")))
    lora_scale = _f_float(lora.get("scale", spec.get("scale")))
    lora_dropout = _f_float(lora.get("dropout", spec.get("dropout")))

    quantization = _detect_quantization(spec)

    model_data: list[tuple[str, str]] = [
        ("Model", model_label),
        ("Dataset", _f_optional_str(spec.get("data"))),
        ("Adapter path", _f_optional_str(spec.get("adapter_path"))),
    ]

    algorithm: list[tuple[str, str]] = [
        ("Mode", _f_optional_str(spec.get("train_mode"))),
        ("Train type", _f_optional_str(spec.get("train_type"))),
        ("Optimizer", _f_optional_str(spec.get("optimizer"))),
        ("Quantization", quantization),
    ]
    if spec.get("reference_model_path"):
        algorithm.append(
            ("Reference model", _f_optional_str(spec.get("reference_model_path")))
        )
    if spec.get("judge"):
        algorithm.append(("Judge", _f_optional_str(spec.get("judge"))))

    optimisation: list[tuple[str, str]] = [
        ("Learning rate", _f_float(spec.get("learning_rate"))),
        ("LR schedule", _format_lr_schedule(spec)),
        ("Batch size", _f_int(spec.get("batch_size"))),
        ("Grad accumulation", _f_int(spec.get("gradient_accumulation_steps"))),
        ("Iters / epochs", _format_iters(spec)),
        ("Val batches", _f_int(spec.get("val_batches"))),
        ("Max seq length", _f_int(spec.get("max_seq_length"))),
    ]

    lora_block: list[tuple[str, str]] = [
        ("Rank", lora_rank),
        ("Scale", lora_scale),
        ("Dropout", lora_dropout),
    ]

    reporting: list[tuple[str, str]] = [
        ("Steps per report", _f_int(spec.get("steps_per_report"))),
        ("Steps per eval", _f_int(spec.get("steps_per_eval"))),
        ("Save every", _f_int(spec.get("save_every"))),
    ]

    toggles: list[tuple[str, str]] = [
        ("Grad checkpoint", _f_bool(spec.get("grad_checkpoint"))),
        ("Efficient long context", _f_bool(spec.get("efficient_long_context"))),
        ("Mask prompt", _f_bool(spec.get("mask_prompt"))),
        ("Fuse", _f_bool(spec.get("fuse"))),
    ]

    groups: list[tuple[str, list[tuple[str, str]]]] = [
        ("Model & Data", model_data),
        ("Algorithm", algorithm),
        ("Optimisation", optimisation),
        ("LoRA Parameters", lora_block),
        ("Reporting & Saving", reporting),
        ("Toggles", toggles),
    ]

    # Algorithm-specific extras. The writer's run_spec.json has flat
    # keys for everything, so we just check the train_mode and pull
    # the relevant ones when they exist.
    train_mode = clean_string(spec.get("train_mode"))
    extras: list[tuple[str, str]] = []
    if "beta" in spec:
        extras.append(("β (beta)", _f_float(spec.get("beta"))))
    if "dpo_cpo_loss_type" in spec:
        extras.append(("Loss type", _f_optional_str(spec.get("dpo_cpo_loss_type"))))
    if "delta" in spec:
        extras.append(("δ (delta)", _f_float(spec.get("delta"))))
    if "alpha" in spec and train_mode in {"online_dpo", "xpo", "rlhf_reinforce", "ppo"}:
        extras.append(("α (alpha)", _f_optional_str(spec.get("alpha"))))
    if "reward_scaling" in spec:
        extras.append(("Reward scaling", _f_float(spec.get("reward_scaling"))))
    if train_mode == "grpo":
        for key, label in [
            ("group_size", "Group size"),
            ("epsilon", "ε (epsilon)"),
            ("epsilon_high", "ε high"),
            ("grpo_loss_type", "Loss type"),
            ("max_completion_length", "Max completion length"),
            ("temperature", "Temperature"),
            ("top_p", "Top-p"),
            ("top_k", "Top-k"),
            ("min_p", "Min-p"),
        ]:
            if key in spec:
                extras.append((label, _format_value(key, spec.get(key))))
    if extras:
        groups.append(("Preference / RL / GRPO", extras))

    if spec.get("qat_enable"):
        qat = [
            ("Bits", _f_int(spec.get("qat_bits"))),
            ("Group size", _f_int(spec.get("qat_group_size"))),
            ("Start step", _f_int(spec.get("qat_start_step"))),
            ("Interval", _f_int(spec.get("qat_interval"))),
        ]
        groups.append(("Quantization-Aware Training", qat))

    # Drop rows whose value is the placeholder dash AND we have a
    # choice of skipping. For required fields (model, dataset) we keep
    # the dash so the table is still complete.
    return groups


def _format_value(key: str, raw: Any) -> str:
    if key in {"epsilon_high", "grpo_loss_type", "reward_weights", "reward_functions"}:
        return _f_optional_str(raw)
    if key in {"temperature", "top_p", "min_p"}:
        return _f_float(raw)
    if key in {"group_size", "max_completion_length", "top_k"}:
        return _f_int(raw)
    if key == "epsilon":
        return _f_float(raw)
    return _f_str(raw)


def _detect_quantization(spec: dict[str, Any]) -> str:
    if spec.get("load_in_4bits"):
        return "4-bit"
    if spec.get("load_in_6bits"):
        return "6-bit"
    if spec.get("load_in_8bits"):
        return "8-bit"
    if spec.get("load_in_mxfp4"):
        return "MXFP4"
    return "none"


def _format_lr_schedule(spec: dict[str, Any]) -> str:
    schedule = spec.get("lr_schedule")
    if not schedule:
        return "constant"
    name = clean_string(schedule.get("name"))
    if name == "constant":
        return "constant"
    warmup = _f_int(schedule.get("warmup"))
    init = _f_float(schedule.get("warmup_init"))
    args = schedule.get("arguments")
    final = "—"
    if isinstance(args, list) and len(args) >= 3:
        final = _f_float(args[2])
    return f"cosine decay (warmup {warmup}, init {init}, final {final})"


def _format_iters(spec: dict[str, Any]) -> str:
    epochs = spec.get("epochs")
    iters = spec.get("iters")
    if isinstance(epochs, (int, float)) and epochs and int(epochs) > 0:
        return f"{int(epochs)} epochs"
    if isinstance(iters, (int, float)) and iters:
        return f"{int(iters)} iters"
    return "—"


# MARK: - Hugging Face card metadata
#
# Hugging Face reads model-card metadata only when the README begins
# with a YAML front matter block. Keep this writer dependency-free so
# uploads still work in a minimal Python environment.


def _yaml_quote(value: Any) -> str:
    text = clean_string(value)
    if not text:
        return '""'
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _yaml_list(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        text = clean_string(value)
        if text and text not in cleaned:
            cleaned.append(text)
    if not cleaned:
        return ["[]"]
    return [f"- {_yaml_quote(value)}" for value in cleaned]


def _metadata_tags(kind: str, spec: dict[str, Any] | None) -> list[str]:
    tags = ["mlx", "mlx-lm-lora", "mlx-lora-studio"]
    if kind == "model":
        tags.extend(["lora", "text-generation"])
    elif kind == "synthetic dataset":
        tags.extend(["synthetic", "instruction-tuning"])
    if spec is not None:
        train_mode = clean_string(spec.get("train_mode"))
        train_type = clean_string(spec.get("train_type"))
        synthetic_kind = clean_string(spec.get("kind"))
        if train_mode:
            tags.append(train_mode)
        if train_type:
            tags.append(train_type)
        if synthetic_kind:
            tags.append(synthetic_kind)
            tags.append(f"synthetic-{synthetic_kind}")
        if spec.get("qat_enable"):
            tags.append("qat")
    return tags


def render_hf_metadata_block(kind: str, spec: dict[str, Any] | None = None) -> str:
    """Return a valid Hugging Face README YAML metadata block.

    HF requires this block to be the first bytes of the README. We keep
    values conservative: known HF card keys plus generated tags, while
    leaving license explicit-but-unspecified because the upstream base
    model/dataset licenses still govern redistribution.
    """
    tags = _metadata_tags(kind, spec)
    lines: list[str] = ["---"]
    if kind == "model":
        lines += [
            "library_name: mlx",
            "pipeline_tag: text-generation",
            "license: other",
            "tags:",
            *_yaml_list(tags),
        ]
        if spec is not None:
            base_model = clean_string(spec.get("model"))
            dataset = clean_string(spec.get("data"))
            if base_model:
                lines += ["base_model:", *_yaml_list([base_model])]
            if dataset:
                lines += ["datasets:", *_yaml_list([dataset])]
    else:
        lines += [
            "license: other",
            "task_categories:",
            "- text-generation",
            "tags:",
            *_yaml_list(tags),
        ]
    lines += ["---", ""]
    return "\n".join(lines)


def render_settings_block(spec: dict[str, Any]) -> str:
    groups = _settings_rows(spec)
    lines: list[str] = ["## Training Settings", ""]
    for title, rows in groups:
        # Drop rows where the value is the "—" placeholder. Optional
        # fields that the user never set shouldn't pad the table.
        cleaned = [(label, value) for label, value in rows if value and value != "—"]
        if not cleaned:
            continue
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        for label, value in cleaned:
            # Escape pipe characters so a value containing `|` doesn't
            # break the table.
            safe = str(value).replace("|", "\\|")
            lines.append(f"| {label} | {safe} |")
        lines.append("")
    return "\n".join(lines).rstrip()


# MARK: - Metrics block
#
# Builds a summary section from `metrics.json`. We pull loss / val_loss
# for the sparkline, plus a small "key metrics at a glance" table of
# the most recent values for the ~10 most-reported keys.


def _collect_metric_series(metrics: list[dict[str, Any]]) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    for entry in metrics:
        values = entry.get("values") if isinstance(entry, dict) else None
        if not isinstance(values, dict):
            continue
        for key, raw in values.items():
            try:
                series.setdefault(key, []).append(float(raw))
            except (TypeError, ValueError):
                continue
    return series


def render_metrics_block(
    metrics: list[dict[str, Any]],
    loss_image_filename: str | None = None,
) -> str:
    series = _collect_metric_series(metrics)
    if not series:
        return ""

    train_loss = series.get("loss", [])
    val_loss = series.get("val_loss", [])

    lines: list[str] = ["## Training Metrics", ""]

    # Headline numbers — the user can see the final loss without
    # scrolling through a table.
    headline: list[str] = []
    if train_loss:
        headline.append(
            f"- **Final train loss:** `{train_loss[-1]:.4f}` (min `{min(train_loss):.4f}`)"
        )
    if val_loss:
        headline.append(
            f"- **Final val loss:** `{val_loss[-1]:.4f}` (min `{min(val_loss):.4f}`)"
        )
    if "learning_rate" in series and series["learning_rate"]:
        headline.append(
            f"- **Final learning rate:** `{series['learning_rate'][-1]:.3e}`"
        )
    if "peak_mem" in series and series["peak_mem"]:
        headline.append(f"- **Peak memory:** `{max(series['peak_mem']):.2f} GB`")
    if headline:
        lines.extend(headline)
        lines.append("")

    # Loss curve. Image takes priority; the unicode sparkline is the
    # always-available fallback (and ships even when the image does,
    # so screen readers see something useful).
    if loss_image_filename is not None:
        lines += [
            "### Loss Curve",
            "",
            f"![Training and validation loss]({loss_image_filename})",
            "",
        ]
    spark_train = render_loss_sparkline(train_loss)
    spark_val = render_loss_sparkline(val_loss)
    if spark_train or spark_val:
        lines.append("### Sparkline")
        lines.append("")
        if spark_train:
            lines.append(f"- Train loss: `{spark_train}`")
        if spark_val:
            lines.append(f"- Validation loss: `{spark_val}`")
        lines.append("")

    # Key-metrics table — pick the top series by sample count so the
    # most-reported metrics (loss, lr, it/s, tok/s, peak_mem) are the
    # ones that get a row, and clip to the top 10.
    ranked = sorted(series.items(), key=lambda pair: -len(pair[1]))
    top = ranked[:10]
    lines.append("### Key Metrics")
    lines.append("")
    lines.append("| Metric | Latest | Min | Max | N |")
    lines.append("|---|---|---|---|---|")
    for key, values in top:
        latest = values[-1]
        if abs(latest) >= 1000 or (abs(latest) < 0.01 and latest != 0):
            latest_str = f"{latest:.2e}"
            min_str = f"{min(values):.2e}"
            max_str = f"{max(values):.2e}"
        else:
            latest_str = f"{latest:.4f}"
            min_str = f"{min(values):.4f}"
            max_str = f"{max(values):.4f}"
        lines.append(
            f"| `{key}` | {latest_str} | {min_str} | {max_str} | {len(values)} |"
        )
    lines.append("")
    return "\n".join(lines).rstrip()


# MARK: - Synthetic dataset cards


def _record_text_values(record: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ["prompt", "completion", "chosen", "rejected", "text"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    values.append(content.strip())
    return values


def _estimate_tokens(text: str) -> int:
    # Tokenizers differ by model. This card uses a transparent rough
    # estimate so users still get scale without making uploads depend
    # on a specific tokenizer package or model download.
    return max(1, math.ceil(len(text) / 4)) if text else 0


def _truncate_markdown(text: str, limit: int = 700) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _profile_jsonl(path: Path, sample_limit: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    rows = 0
    fields: set[str] = set()
    token_estimate = 0
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            rows += 1
            fields.update(str(k) for k in record.keys())
            text_parts = _record_text_values(record)
            token_estimate += sum(_estimate_tokens(part) for part in text_parts)
            if len(samples) < sample_limit:
                samples.append(record)
    return (
        {
            "rows": rows,
            "fields": sorted(fields),
            "estimated_tokens": token_estimate,
            "samples": samples,
            "source_file": path.name,
        }
        if rows
        else None
    )


def _profile_parquet(path: Path, sample_limit: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None
    try:
        table = pq.read_table(path)
        rows = table.num_rows
        fields = table.column_names
        samples = table.slice(0, sample_limit).to_pylist()
        token_estimate = 0
        text_columns = [
            name
            for name in fields
            if name in {"prompt", "completion", "chosen", "rejected", "text"}
        ]
        for name in text_columns:
            column = table.column(name).to_pylist()
            token_estimate += sum(
                _estimate_tokens(v) for v in column if isinstance(v, str)
            )
        return {
            "rows": rows,
            "fields": fields,
            "estimated_tokens": token_estimate,
            "samples": [s for s in samples if isinstance(s, dict)],
            "source_file": path.name,
        }
    except Exception:
        return None


def profile_dataset(folder: Path, sample_limit: int = 3) -> dict[str, Any] | None:
    jsonl_profile = _profile_jsonl(folder / "output_full.jsonl", sample_limit)
    if jsonl_profile is not None:
        return jsonl_profile
    data_dir = folder / "data"
    for parquet_path in sorted(data_dir.glob("*.parquet")) if data_dir.exists() else []:
        profile = _profile_parquet(parquet_path, sample_limit)
        if profile is not None:
            return profile
    return None


def _synthetic_models(spec: dict[str, Any] | None) -> list[tuple[str, str]]:
    if spec is None:
        return []
    kind = clean_string(spec.get("kind"))
    if kind == "sft":
        return [("Generator model", clean_string(spec.get("model")))]
    if kind == "dpo":
        return [
            ("Base / rejected model", clean_string(spec.get("base_model"))),
            ("Teacher / chosen model", clean_string(spec.get("teacher_model"))),
        ]
    return []


def render_dataset_block(
    dataset_profile: dict[str, Any] | None,
    synthetic_spec: dict[str, Any] | None,
) -> str:
    lines: list[str] = ["## Dataset Details", ""]
    kind = clean_string(synthetic_spec.get("kind")) if synthetic_spec else ""
    if kind:
        lines.append(f"- **Generation type:** `{kind.upper()}`")
    if synthetic_spec is not None:
        source = clean_string(synthetic_spec.get("dataset_path"))
        if source:
            lines.append(f"- **Source dataset:** `{source}`")
        target = clean_string(synthetic_spec.get("dpo_generation_target"))
        if target:
            lines.append(f"- **DPO generation target:** `{target}`")
        backend = clean_string(synthetic_spec.get("backend"))
        if backend:
            lines.append(f"- **Backend:** `{backend}`")
        models = [
            (label, value)
            for label, value in _synthetic_models(synthetic_spec)
            if value
        ]
        for label, value in models:
            lines.append(f"- **{label}:** `{value}`")
    if dataset_profile is not None:
        rows = dataset_profile.get("rows")
        if rows is not None:
            lines.append(f"- **Samples:** `{int(rows):,}`")
        estimated_tokens = dataset_profile.get("estimated_tokens")
        if estimated_tokens is not None:
            lines.append(f"- **Estimated tokens:** `~{int(estimated_tokens):,}`")
        source_file = clean_string(dataset_profile.get("source_file"))
        if source_file:
            lines.append(f"- **Profiled file:** `{source_file}`")
        fields = dataset_profile.get("fields")
        if isinstance(fields, list) and fields:
            rendered = ", ".join(f"`{field}`" for field in fields)
            lines.append(f"- **Columns:** {rendered}")
    lines += [
        "",
        "Token count is estimated from text length and is intended as a quick dataset-scale signal, not tokenizer-exact accounting.",
        "",
    ]

    samples = dataset_profile.get("samples") if dataset_profile else None
    if isinstance(samples, list) and samples:
        lines += ["### Samples", ""]
        for index, sample in enumerate(samples, start=1):
            if not isinstance(sample, dict):
                continue
            lines.append(f"<details><summary>Sample {index}</summary>")
            lines.append("")
            for key in ["prompt", "completion", "chosen", "rejected", "text"]:
                value = sample.get(key)
                if isinstance(value, str) and value.strip():
                    lines.append(f"**{key}**")
                    lines.append("")
                    lines.append(_truncate_markdown(value))
                    lines.append("")
            messages = sample.get("messages")
            if isinstance(messages, list):
                lines.append("**messages**")
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(messages[:6], ensure_ascii=False, indent=2))
                lines.append("```")
                lines.append("")
            lines.append("</details>")
            lines.append("")
    return "\n".join(lines).rstrip()


# MARK: - README writer


def write_readme(
    folder: Path, repo_id: str, kind: str, upload_kind: str | None = None
) -> None:
    """Write the README that gets uploaded alongside the model (or
    dataset) folder. The model-side README is enriched with the run's
    training settings + metrics when we can find a matching run
    folder; the dataset-side README stays minimal."""
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    title = repo_id.split("/", 1)[1]

    # Resolve the run folder *before* writing the README so we can
    # hand its data to the metrics block builder. The dataset path
    # never has a parent run folder, so this only enriches the model
    # README.
    run_folder = resolve_run_folder(folder) if kind == "model" else None
    synthetic_run_folder = (
        resolve_synthetic_run_folder(folder) if kind == "synthetic dataset" else None
    )
    spec = load_spec(run_folder) if run_folder else None
    synthetic_spec = (
        load_synthetic_spec(synthetic_run_folder) if synthetic_run_folder else None
    )
    card_spec = spec if spec is not None else synthetic_spec
    metrics = load_metrics(run_folder) if run_folder else []
    dataset_profile = profile_dataset(folder) if kind == "synthetic dataset" else None
    if kind == "synthetic dataset" and synthetic_spec is not None:
        (folder / "synthetic_spec.json").write_text(
            json.dumps(synthetic_spec, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    # Render a loss-curve PNG when matplotlib is importable. We write
    # the image to the model folder (it travels with the upload) and
    # the README embeds it via a relative path. Falls back to the
    # unicode sparkline silently when matplotlib is missing.
    loss_image_filename: str | None = None
    if run_folder is not None and spec is not None:
        png_path = folder / LOSS_CURVE_IMAGE_NAME
        train_loss = [
            float(m["values"]["loss"])
            for m in metrics
            if isinstance(m, dict)
            and isinstance(m.get("values"), dict)
            and "loss" in m["values"]
        ]
        val_loss = [
            float(m["values"]["val_loss"])
            for m in metrics
            if isinstance(m, dict)
            and isinstance(m.get("values"), dict)
            and "val_loss" in m["values"]
        ]
        if render_loss_png(train_loss, val_loss, png_path):
            loss_image_filename = LOSS_CURVE_IMAGE_NAME
            studio_log(f"Rendered {LOSS_CURVE_IMAGE_NAME}")

    lines: list[str] = [
        render_hf_metadata_block(kind, card_spec),
        f"# {title}",
        "",
        f"> {README_TAGLINE}",
        "",
        f"![asset](https://img.shields.io/badge/asset-{kind.replace(' ', '%20')}-orange)"
        f"![upload](https://img.shields.io/badge/upload-MLX%20LoRA%20Studio-blue)",
        "",
        "## Overview",
        "",
        f"- **Repository:** `{repo_id}`",
        f"- **Asset type:** {kind}",
        f"- **Created at:** {created_at}",
    ]
    if upload_kind:
        lines.append(f"- **Model upload mode:** {upload_kind}")
    if synthetic_spec is not None:
        synthetic_kind = clean_string(synthetic_spec.get("kind"))
        if synthetic_kind:
            lines.append(f"- **Synthetic data type:** {synthetic_kind.upper()}")
        for label, value in _synthetic_models(synthetic_spec):
            if value:
                lines.append(f"- **{label}:** `{value}`")
    if dataset_profile is not None:
        rows = dataset_profile.get("rows")
        if rows is not None:
            lines.append(f"- **Samples:** `{int(rows):,}`")
        estimated_tokens = dataset_profile.get("estimated_tokens")
        if estimated_tokens is not None:
            lines.append(f"- **Estimated tokens:** `~{int(estimated_tokens):,}`")
    if spec is not None:
        model_short = clean_string(spec.get("model"))
        if model_short:
            lines.append(f"- **Base model:** `{model_short}`")
        train_mode = clean_string(spec.get("train_mode"))
        if train_mode:
            lines.append(f"- **Algorithm:** {train_mode}")
        data = clean_string(spec.get("data"))
        if data:
            lines.append(f"- **Dataset:** `{data}`")
        if "iters" in spec or "epochs" in spec:
            lines.append(f"- **Training length:** {_format_iters(spec)}")
        train_loss = [
            m
            for m in metrics
            if isinstance(m, dict)
            and isinstance(m.get("values"), dict)
            and "loss" in m["values"]
        ]
        if train_loss:
            final = float(train_loss[-1]["values"]["loss"])
            lines.append(f"- **Final train loss:** `{final:.4f}`")
        val_loss = [
            m
            for m in metrics
            if isinstance(m, dict)
            and isinstance(m.get("values"), dict)
            and "val_loss" in m["values"]
        ]
        if val_loss:
            final = float(val_loss[-1]["values"]["val_loss"])
            lines.append(f"- **Final validation loss:** `{final:.4f}`")

    lines += [
        "",
        "This repository was prepared by MLX LoRA Studio from local training outputs.",
        "",
    ]

    # Training settings + metrics blocks (only when we have a spec to
    # show). The metrics block is always included when we have a run
    # folder, even if the spec itself didn't decode — the user might
    # have started a run that produced metrics but where the spec was
    # corrupted.
    if spec is not None:
        lines.append(render_settings_block(spec))
        lines.append("")
    if kind == "synthetic dataset":
        lines.append(render_dataset_block(dataset_profile, synthetic_spec))
        lines.append("")
    if run_folder is not None and metrics:
        lines.append(
            render_metrics_block(metrics, loss_image_filename=loss_image_filename)
        )
        lines.append("")

    # Reproducibility / license footer.
    lines += [
        "## Reproducibility",
        "",
        f"The full `{('run_spec.json' if run_folder is not None else 'synthetic_spec.json' if synthetic_run_folder is not None else 'spec.json')}` used to launch this run is included in the repository. "
        "Re-running the same spec on the same model(s), source dataset, and generation settings should reproduce an equivalent artifact "
        "(up to sampling and kernel-level non-determinism).",
        "",
        "## About",
        "",
        f"**{README_AUTHOR}** — `{README_STAMP}`",
        "",
        f"MLX LoRA Studio is a SwiftUI desktop app for fine-tuning open language models on Apple Silicon with the [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) trainer. "
        f"Curated by {README_AUTHOR}.",
        "",
        "## License",
        "",
        "The license of the upstream base model(s), source dataset, generated dataset, and any included tokenizer files applies. "
        "Check the source model and dataset cards before redistribution or downstream training.",
        "",
    ]

    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")
    studio_log(f"Created README.md in {folder}")


def adapter_allow_patterns(folder: Path) -> list[str]:
    patterns = [
        "README.md",
        "adapter_config.json",
        "adapters*.safetensors",
        "*adapters.safetensors",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
        "generation_config.json",
        # Loss-curve image produced when matplotlib is available in
        # the upload environment. Including it here means the README
        # image and the README itself travel together even in
        # adapters-only mode.
        LOSS_CURVE_IMAGE_NAME,
    ]
    # Include common nested tokenizer assets without accidentally pulling
    # large model shards from arbitrary subdirectories.
    if (
        any((folder / "tokenizer").glob("*"))
        if (folder / "tokenizer").exists()
        else False
    ):
        patterns.append("tokenizer/*")
    return patterns


def upload_folder(
    api: Any,
    create_repo: Any,
    folder: Path,
    repo_id: str,
    repo_type: str,
    private: bool,
    commit_message: str,
    allow_patterns: list[str] | None = None,
) -> None:
    create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    studio_log(f"Uploading {folder} to {repo_id} ({repo_type})")
    kwargs: dict[str, Any] = {
        "folder_path": str(folder),
        "repo_id": repo_id,
        "repo_type": repo_type,
        "commit_message": commit_message,
    }
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    api.upload_folder(**kwargs)
    repo_path = "datasets/" + repo_id if repo_type == "dataset" else repo_id
    studio_log(f"Upload complete: https://huggingface.co/{repo_path}")


def run(spec: dict[str, Any]) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Add one in Settings before uploading."
        )

    HfApi, create_repo = require_hub()
    api = HfApi(token=token)

    upload_target = clean_string(spec.get("upload_target")) or "all"
    upload_kind = clean_string(spec.get("model_upload_kind")) or "adaptersOnly"
    private = bool(spec.get("private"))
    commit_message = (
        clean_string(spec.get("commit_message")) or "Upload from MLX LoRA Studio"
    )
    upload_model = upload_target in {"model", "all"}
    upload_dataset = upload_target in {"dataset", "all"} and (
        upload_target == "dataset" or bool(spec.get("upload_synthetic_dataset"))
    )

    if upload_target not in {"model", "dataset", "all"}:
        raise ValueError(f"Unsupported upload target: {upload_target}")
    if upload_kind not in {"adaptersOnly", "mergedWeights"}:
        raise ValueError(f"Unsupported model upload kind: {upload_kind}")
    if not upload_model and not upload_dataset:
        raise ValueError("Nothing selected to upload.")

    if upload_model:
        model_repo = validate_repo(clean_string(spec.get("model_repo")), "Model repo")
        model_path = validate_path(
            clean_string(spec.get("local_model_path")), "Local model path"
        )
        write_readme(model_path, model_repo, "model", upload_kind)
        allow_patterns = (
            adapter_allow_patterns(model_path)
            if upload_kind == "adaptersOnly"
            else None
        )
        upload_folder(
            api,
            create_repo,
            model_path,
            model_repo,
            "model",
            private,
            commit_message,
            allow_patterns=allow_patterns,
        )

    if upload_dataset:
        dataset_repo = validate_repo(
            clean_string(spec.get("dataset_repo")), "Dataset repo"
        )
        dataset_path = validate_path(
            clean_string(spec.get("local_dataset_path")), "Local dataset path"
        )
        write_readme(dataset_path, dataset_repo, "synthetic dataset")
        upload_folder(
            api,
            create_repo,
            dataset_path,
            dataset_repo,
            "dataset",
            private,
            commit_message,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload MLX LoRA Studio outputs to Hugging Face Hub."
    )
    parser.add_argument(
        "--spec", required=True, help="Path to the Studio JSON upload spec."
    )
    args = parser.parse_args()
    with open(args.spec, "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    try:
        run(spec)
    except Exception as exc:
        studio_log(f"Upload failed: {type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
