#!/usr/bin/env python3
"""App-owned training runner with notebook-style per-algorithm pipelines."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import resource
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx_lm.tuner.callbacks import TrainingCallback, WandBCallback
from mlx_lm.tuner.utils import build_schedule, load_adapters, print_trainable_parameters
from mlx_lm_lora.train import (
    CONFIG_DEFAULTS,
    build_lora_config,
    evaluate_model,
    load_judge_model,
    load_reference_model,
    load_reward_functions_from_file,
)
from mlx_lm_lora.trainer.cpo_trainer import CPOTrainingArgs, train_cpo
from mlx_lm_lora.trainer.datasets import CacheDataset, load_dataset
from mlx_lm_lora.trainer.dpo_trainer import DPOTrainingArgs, train_dpo
from mlx_lm_lora.trainer.grpo_reward_functions import (
    get_default_reward_functions,
    get_reward_function,
    list_available_reward_functions,
)
from mlx_lm_lora.trainer.grpo_trainer import GRPOTrainingArgs, train_grpo
from mlx_lm_lora.trainer.online_dpo_trainer import (
    OnlineDPOTrainingArgs,
    train_online_dpo,
)
from mlx_lm_lora.trainer.orpo_trainer import ORPOTrainingArgs, train_orpo
from mlx_lm_lora.trainer.ppo_trainer import PPOTrainingArgs, train_ppo
from mlx_lm_lora.trainer.rlhf_reinforce_trainer import (
    RLHFReinforceTrainingArgs,
    train_rlhf_reinforce,
)
from mlx_lm_lora.trainer.sft_trainer import SFTTrainingArgs, train_sft
from mlx_lm_lora.trainer.xpo_trainer import XPOTrainingArgs, train_xpo
from mlx_lm_lora.utils import (
    from_pretrained,
    save_pretrained_merged,
    save_pretrained_merged_vision,
    save_to_lmstudio_merged,
)

REFERENCE_MODES = {"dpo", "grpo", "online_dpo", "ppo", "rlhf_reinforce", "xpo"}
JUDGE_MODES = {"online_dpo", "ppo", "rlhf_reinforce", "xpo"}
STUDIO_OUT = sys.stdout
OPTIMIZER_CLASSES = {
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
    "adadelta": optim.AdaDelta,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "adamax": optim.Adamax,
    "lion": optim.Lion,
    "adafactor": optim.Adafactor,
    "muon": optim.Muon,
}


class ResourceGuard:
    """Abort before a training run consumes unsafe unified-memory headroom."""

    def __init__(self) -> None:
        self.enabled = os.environ.get(
            "MLX_LORA_STUDIO_RESOURCE_GUARD", "1"
        ).lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.total_memory = self._total_memory()
        fraction = self._float_env("MLX_LORA_STUDIO_MEMORY_LIMIT_FRACTION", 0.78)
        self.fraction = min(max(fraction, 0.10), 0.98)
        self.max_bytes = self._limit_bytes()
        self._last_log_bucket: int | None = None

    def describe(self) -> str | None:
        if not self.enabled or not self.max_bytes:
            return None
        return (
            f"Resource guard armed at {self.max_bytes / 1e9:.1f} GB "
            f"({self.fraction:.0%} of system memory)."
        )

    def check(self, context: str) -> None:
        if not self.enabled or not self.max_bytes:
            return
        used = max(self._process_peak_bytes(), self._mlx_peak_bytes())
        if used <= 0:
            return
        if used > self.max_bytes * 0.72:
            self.release_caches()
            used = max(self._process_peak_bytes(), self._mlx_peak_bytes())
        bucket = int(used / 1_000_000_000)
        if bucket != self._last_log_bucket and used > self.max_bytes * 0.75:
            self._last_log_bucket = bucket
            studio_log(
                f"Resource guard: peak memory {used / 1e9:.1f} GB / "
                f"{self.max_bytes / 1e9:.1f} GB while {context}."
            )
        if used >= self.max_bytes:
            raise RuntimeError(
                "Resource guard cancelled the run before memory pressure could "
                f"destabilize macOS ({used / 1e9:.1f} GB peak >= "
                f"{self.max_bytes / 1e9:.1f} GB limit). Lower batch size, max "
                "sequence length, rank, or use stronger quantization before retrying."
            )

    @staticmethod
    def release_caches() -> None:
        with contextlib.suppress(Exception):
            mx.clear_cache()
        reset_peak = getattr(mx, "reset_peak_memory", None)
        if callable(reset_peak):
            with contextlib.suppress(Exception):
                reset_peak()

    def _limit_bytes(self) -> int:
        explicit_gb = self._float_env("MLX_LORA_STUDIO_MEMORY_LIMIT_GB", 0.0)
        if explicit_gb > 0:
            return int(explicit_gb * 1_000_000_000)
        if self.total_memory:
            return int(self.total_memory * self.fraction)
        return 0

    @staticmethod
    def _float_env(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _total_memory() -> int:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, OSError, ValueError):
            return 0

    @staticmethod
    def _process_peak_bytes() -> int:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(peak)
        return int(peak * 1024)

    @staticmethod
    def _mlx_peak_bytes() -> int:
        try:
            return int(mx.get_peak_memory())
        except Exception:
            return 0


class QuietVendorOutput:
    """Swallow trainer banners, tqdm, ANSI art, and other raw CLI chatter."""

    def write(self, _text: str) -> int:
        return len(_text)

    def flush(self) -> None:
        pass


class TeeCapture:
    """Forward output to the app terminal while keeping a copy for parsing."""

    def __init__(self, stream=STUDIO_OUT) -> None:
        self.stream = stream
        self.buffer = io.StringIO()

    def write(self, text: str) -> int:
        self.buffer.write(text)
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()


@contextlib.contextmanager
def quiet_vendor_output():
    sink = QuietVendorOutput()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


def studio_log(message: str) -> None:
    print(f"[Studio] {message}", file=STUDIO_OUT, flush=True)


def _positive_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return number


def _non_negative_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if number < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return number


def _positive_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number.") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return number


class StudioCallback(TrainingCallback):
    """Emit structured metrics plus compact app-terminal progress lines."""

    def __init__(self, stream=STUDIO_OUT, guard: ResourceGuard | None = None):
        self.stream = stream
        self.guard = guard

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.guard is not None:
            self.guard.check(f"reporting {event} metrics")
        serializable = {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in payload.items()
        }
        print(
            "@@studio_metric "
            + json.dumps({"event": event, **serializable}, sort_keys=True),
            file=self.stream,
            flush=True,
        )
        line = self._format_line(event, serializable)
        if line:
            print(line, file=self.stream, flush=True)

    def _format_line(self, event: str, payload: dict[str, Any]) -> str:
        step = payload.get("iteration", 0)
        parts = [f"Iter {step}", event.title()]
        loss = payload.get(
            "train_loss", payload.get("val_loss", payload.get("test_loss"))
        )
        if loss is not None:
            label = "loss"
            if event == "validation":
                label = "val loss"
            elif event == "test":
                label = "test loss"
            parts.append(f"{label} {float(loss):.3f}")
        lr = payload.get("learning_rate")
        if lr is not None:
            parts.append(f"lr {float(lr):.2e}")
        it_s = payload.get("iterations_per_second")
        if it_s is not None:
            parts.append(f"{float(it_s):.2f} it/s")
        tok_s = payload.get("tokens_per_second")
        if tok_s is not None:
            parts.append(f"{float(tok_s):.0f} tok/s")
        peak = payload.get("peak_memory")
        if peak is not None:
            parts.append(f"{float(peak):.2f} GB")
        return " | ".join(parts)

    def on_train_loss_report(self, train_info: dict[str, Any]) -> None:
        self._emit("train", train_info)

    def on_val_loss_report(self, val_info: dict[str, Any]) -> None:
        self._emit("validation", val_info)

    def on_test_loss_report(self, test_info: dict[str, Any]) -> None:
        self._emit("test", test_info)


def _dataset_len(dataset: Any) -> int | None:
    try:
        return len(dataset)
    except TypeError:
        return None


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)


def _first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _test_metrics_from_output(output: str, iteration: int) -> dict[str, Any]:
    plain = _strip_ansi(output)
    metrics: dict[str, Any] = {"iteration": iteration}
    loss = _first_float(r"\bLoss:\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", plain)
    if loss is not None:
        metrics["test_loss"] = loss
    ppl = _first_float(r"\bPerplexity:\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", plain)
    if ppl is not None:
        metrics["test_ppl"] = ppl
    tokens = _first_float(r"\bTokens:\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", plain)
    if tokens is not None:
        metrics["test_tokens"] = tokens
    rewards = re.search(
        r"\bRewards:\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?),\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)",
        plain,
        flags=re.IGNORECASE,
    )
    if rewards:
        metrics["test_chosen_reward"] = float(rewards.group(1))
        metrics["test_rejected_reward"] = float(rewards.group(2))
    return metrics


def _normalize_spec(spec: dict[str, Any]) -> SimpleNamespace:
    args = dict(CONFIG_DEFAULTS)
    args.update(spec)
    if args.get("data"):
        args["data"] = _prepare_local_dataset_for_trainer(str(args["data"]))
    args["train"] = True
    args["config"] = None
    args["optimizer_config"] = args.get("optimizer_config") or {
        "sgd": {},
        "rmsprop": {},
        "adagrad": {},
        "adadelta": {},
        "adam": {},
        "adamw": {},
        "adamax": {},
        "lion": {},
        "adafactor": {},
        "muon": {},
    }
    args["judge_config"] = args.get("judge_config") or {}
    args["hf_dataset"] = args.get("hf_dataset") or None
    args["lr_schedule"] = args.get("lr_schedule") or None

    for key in (
        "reference_model_path",
        "resume_adapter_file",
        "reward_weights",
        "reward_functions",
        "reward_functions_file",
        "importance_sampling_level",
        "lm_studio_name",
        "vlm_model",
        "vlm_output_path",
        "epsilon_high",
    ):
        if args.get(key) in (None, ""):
            args[key] = None

    args["model_family"] = args.get("model_family") or "text"
    args["vlm_dequantize"] = bool(args.get("vlm_dequantize", True))
    args["fuse_dequantize"] = bool(args.get("fuse_dequantize", True))
    args["fuse_remove_adapters"] = bool(args.get("fuse_remove_adapters", True))

    args["batch_size"] = _positive_int(args.get("batch_size"), "batch_size")
    args["gradient_accumulation_steps"] = _positive_int(
        args.get("gradient_accumulation_steps"),
        "gradient_accumulation_steps",
    )
    args["max_seq_length"] = _positive_int(args.get("max_seq_length"), "max_seq_length")
    args["steps_per_report"] = _positive_int(
        args.get("steps_per_report"), "steps_per_report"
    )
    args["steps_per_eval"] = _positive_int(args.get("steps_per_eval"), "steps_per_eval")
    args["save_every"] = _positive_int(args.get("save_every"), "save_every")
    args["val_batches"] = _non_negative_int(args.get("val_batches"), "val_batches")
    args["learning_rate"] = _positive_float(args.get("learning_rate"), "learning_rate")
    if args.get("iters") is not None:
        args["iters"] = _positive_int(args.get("iters"), "iters")
    if args.get("epochs") is not None:
        args["epochs"] = _positive_int(args.get("epochs"), "epochs")

    if isinstance(args.get("alpha"), str):
        args["alpha"] = [
            float(value.strip())
            for value in args["alpha"].strip("[]").split(",")
            if value.strip()
        ]
    elif isinstance(args.get("alpha"), (int, float)):
        args["alpha"] = [float(args["alpha"])]
    args["seq_step_size"] = (
        int(args.get("seq_step_size") or 512)
        if args.get("efficient_long_context")
        else None
    )

    return SimpleNamespace(**args)


def _prepare_local_dataset_for_trainer(data: str) -> str:
    """Make Studio synthetic outputs readable by the local JSONL loader.

    The synthetic generators write a human-readable ``output_full.jsonl`` plus
    HF-style parquet shards under ``data/``. The current local trainer loader
    only looks for ``train.jsonl`` / ``valid.jsonl`` / ``test.jsonl`` in the
    selected folder, so previous synthetic runs need a tiny compatibility
    materialization before fine-tuning can start.
    """

    data_path = Path(os.path.expanduser(data))
    if not data_path.exists() or data_path.is_file():
        return data

    candidate_dirs = []
    generated_data_dir = data_path / "generated-data"
    if generated_data_dir.exists():
        generated_data_splits = generated_data_dir / "data"
        if generated_data_splits.exists():
            candidate_dirs.append(generated_data_splits)
        candidate_dirs.append(generated_data_dir)
    nested_data_dir = data_path / "data"
    if nested_data_dir.exists():
        candidate_dirs.append(nested_data_dir)
    candidate_dirs.append(data_path)

    for candidate in candidate_dirs:
        if (candidate / "train.jsonl").exists():
            return str(candidate)

    for candidate in candidate_dirs:
        if _materialize_jsonl_splits(candidate):
            return str(candidate)

    return data


def _materialize_jsonl_splits(folder: Path) -> bool:
    parquet_names = {
        "train": folder / "train-00000-of-00001.parquet",
        "valid": folder / "valid-00000-of-00001.parquet",
        "test": folder / "test-00000-of-00001.parquet",
    }
    if parquet_names["train"].exists():
        try:
            from datasets import Dataset

            for split, parquet_path in parquet_names.items():
                if parquet_path.exists():
                    Dataset.from_parquet(str(parquet_path)).to_json(
                        str(folder / f"{split}.jsonl"),
                        orient="records",
                        lines=True,
                    )
            return (folder / "train.jsonl").exists()
        except Exception as exc:
            studio_log(f"Could not materialize parquet splits for training: {exc}")

    output_full = folder / "output_full.jsonl"
    if output_full.exists():
        train_jsonl = folder / "train.jsonl"
        if not train_jsonl.exists():
            train_jsonl.write_text(
                output_full.read_text(encoding="utf-8"), encoding="utf-8"
            )
        return train_jsonl.exists()

    return False


def _quantization_config(args: SimpleNamespace) -> dict[str, Any] | None:
    if args.load_in_4bits:
        return {"bits": 4, "group_size": 128}
    if args.load_in_6bits:
        return {"bits": 6, "group_size": 128}
    if args.load_in_8bits:
        return {"bits": 8, "group_size": 128}
    if args.load_in_mxfp4:
        return {"bits": 4, "group_size": 32, "mode": "mxfp4"}
    return None


def _optimizer(args: SimpleNamespace):
    optimizer_name = args.optimizer.lower()
    if optimizer_name not in OPTIMIZER_CLASSES:
        supported = ", ".join(sorted(OPTIMIZER_CLASSES))
        raise ValueError(
            f"Unsupported optimizer '{args.optimizer}'. Choose one of: {supported}."
        )
    lr = build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate
    opt_class = OPTIMIZER_CLASSES[optimizer_name]
    opt_config = args.optimizer_config.get(optimizer_name, {})
    return opt_class(learning_rate=lr, **opt_config)


def _resolve_lr_schedule(args: SimpleNamespace) -> None:
    if not args.lr_schedule:
        return
    schedule = dict(args.lr_schedule)
    arguments = list(schedule.get("arguments", []))
    if len(arguments) < 2:
        raise ValueError(
            "lr_schedule.arguments must include at least initial LR and decay steps."
        )

    decay_steps = arguments[1]
    if isinstance(decay_steps, dict) and "iters_fraction" in decay_steps:
        fraction = float(decay_steps["iters_fraction"])
        arguments[1] = max(1, int(args.iters * fraction))

    schedule["arguments"] = arguments
    args.lr_schedule = schedule


def _iters(args: SimpleNamespace, train_set) -> int:
    if args.iters is not None:
        return args.iters
    if args.epochs is None:
        raise ValueError("Either iters or epochs must be provided.")
    if len(train_set) <= 0:
        raise ValueError("Training dataset is empty.")
    batches = math.ceil(len(train_set) / args.batch_size)
    computed = args.epochs * batches
    studio_log(
        f"Calculated {computed} iterations from {args.epochs} epochs "
        f"(dataset size: {len(train_set)}, batch size: {args.batch_size})"
    )
    return computed


def _base_training_kwargs(args: SimpleNamespace, adapter_file: Path) -> dict[str, Any]:
    return {
        "batch_size": args.batch_size,
        "iters": args.iters,
        "val_batches": args.val_batches,
        "steps_per_report": args.steps_per_report,
        "steps_per_eval": args.steps_per_eval,
        "steps_per_save": args.save_every,
        "adapter_file": adapter_file,
        "max_seq_length": args.max_seq_length,
        "grad_checkpoint": args.grad_checkpoint,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }


def _seq_step(args: SimpleNamespace) -> int | None:
    return args.seq_step_size if args.efficient_long_context else None


def _qat_kwargs(args: SimpleNamespace) -> dict[str, Any]:
    return {
        "qat_enable": args.qat_enable,
        "qat_bits": args.qat_bits,
        "qat_group_size": args.qat_group_size,
        "qat_mode": args.qat_mode,
        "qat_start_step": args.qat_start_step,
        "qat_interval": args.qat_interval,
    }


def run_sft(
    args,
    model,
    _tokenizer,
    _ref_model,
    _judge_model,
    _judge_tokenizer,
    opt,
    train_set,
    valid_set,
    adapter_file,
    callback,
):
    train_sft(
        model=model,
        args=SFTTrainingArgs(
            **_base_training_kwargs(args, adapter_file),
            seq_step_size=_seq_step(args),
            **_qat_kwargs(args),
        ),
        optimizer=opt,
        train_dataset=train_set,
        val_dataset=valid_set,
        training_callback=callback,
    )


def run_dpo(
    args,
    model,
    _tokenizer,
    ref_model,
    _judge_model,
    _judge_tokenizer,
    opt,
    train_set,
    valid_set,
    adapter_file,
    callback,
):
    train_dpo(
        model=model,
        ref_model=ref_model,
        optimizer=opt,
        train_dataset=train_set,
        val_dataset=valid_set,
        args=DPOTrainingArgs(
            **_base_training_kwargs(args, adapter_file),
            beta=args.beta,
            loss_type=args.dpo_cpo_loss_type,
            delta=args.delta,
            reference_model_path=args.reference_model_path,
            seq_step_size=_seq_step(args),
            **_qat_kwargs(args),
        ),
        training_callback=callback,
    )


def run_cpo(
    args,
    model,
    _tokenizer,
    _ref_model,
    _judge_model,
    _judge_tokenizer,
    opt,
    train_set,
    valid_set,
    adapter_file,
    callback,
):
    train_cpo(
        model=model,
        optimizer=opt,
        train_dataset=train_set,
        val_dataset=valid_set,
        args=CPOTrainingArgs(
            **_base_training_kwargs(args, adapter_file),
            beta=args.beta,
            loss_type=args.dpo_cpo_loss_type,
            delta=args.delta,
            seq_step_size=_seq_step(args),
            reference_model_path=args.reference_model_path,
            **_qat_kwargs(args),
        ),
        training_callback=callback,
    )


def run_orpo(
    args,
    model,
    _tokenizer,
    _ref_model,
    _judge_model,
    _judge_tokenizer,
    opt,
    train_set,
    valid_set,
    adapter_file,
    callback,
):
    train_orpo(
        model=model,
        optimizer=opt,
        train_dataset=train_set,
        val_dataset=valid_set,
        args=ORPOTrainingArgs(
            **_base_training_kwargs(args, adapter_file),
            beta=args.beta,
            seq_step_size=_seq_step(args),
            reward_scaling=args.reward_scaling,
            **_qat_kwargs(args),
        ),
        training_callback=callback,
    )


def run_grpo(
    args,
    model,
    tokenizer,
    ref_model,
    _judge_model,
    _judge_tokenizer,
    opt,
    train_set,
    valid_set,
    adapter_file,
    callback,
):
    if args.reward_functions_file:
        load_reward_functions_from_file(args.reward_functions_file)

    reward_funcs = get_default_reward_functions()
    if args.reward_functions:
        names = [
            name.strip() for name in args.reward_functions.split(",") if name.strip()
        ]
        try:
            reward_funcs = [get_reward_function(name) for name in names]
            studio_log(f"Using reward functions: {', '.join(names)}")
        except KeyError as exc:
            studio_log(str(exc))
            studio_log(
                f"Available reward functions: {list_available_reward_functions()}"
            )
            raise

    train_grpo(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=opt,
        train_dataset=train_set,
        val_dataset=valid_set,
        reward_funcs=reward_funcs,
        args=GRPOTrainingArgs(
            **_base_training_kwargs(args, adapter_file),
            max_completion_length=args.max_completion_length,
            beta=args.beta,
            group_size=args.group_size,
            epsilon=args.epsilon,
            epsilon_high=args.epsilon_high,
            reference_model_path=args.reference_model_path,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            reward_weights=(
                [float(x) for x in args.reward_weights.strip("[]").split(",")]
                if args.reward_weights
                else None
            ),
            importance_sampling_level=args.importance_sampling_level,
            grpo_loss_type=args.grpo_loss_type,
            **_qat_kwargs(args),
        ),
        training_callback=callback,
    )


def run_online_family(
    args,
    model,
    tokenizer,
    ref_model,
    judge_model,
    judge_tokenizer,
    opt,
    train_set,
    valid_set,
    adapter_file,
    callback,
):
    func, arg_type = {
        "online_dpo": (train_online_dpo, OnlineDPOTrainingArgs),
        "ppo": (train_ppo, PPOTrainingArgs),
        "rlhf_reinforce": (train_rlhf_reinforce, RLHFReinforceTrainingArgs),
        "xpo": (train_xpo, XPOTrainingArgs),
    }[args.train_mode]

    kwargs = {
        **_base_training_kwargs(args, adapter_file),
        "beta": args.beta,
        "reference_model_path": args.reference_model_path,
        "judge": args.judge,
        "max_completion_length": args.max_completion_length,
        **_qat_kwargs(args),
    }
    if args.train_mode in {"online_dpo", "xpo", "ppo"}:
        kwargs["judge_system"] = args.judge_system
    if args.train_mode in {"online_dpo", "xpo"}:
        kwargs.update({"loss_type": args.dpo_cpo_loss_type, "delta": args.delta})
    if args.train_mode in {"online_dpo", "xpo"}:
        kwargs["temperature"] = args.temperature
    if args.train_mode == "ppo":
        kwargs.update(
            {
                "loss_type": args.dpo_cpo_loss_type,
                "delta": args.delta,
                "epsilon": args.epsilon,
                "temperature": args.temperature,
            }
        )
    if args.train_mode == "xpo":
        kwargs["alpha"] = args.alpha

    func(
        model=model,
        tokenizer=tokenizer,
        ref_model=ref_model,
        judge_model=judge_model,
        judge_tokenizer=judge_tokenizer,
        judge_config=args.judge_config,
        optimizer=opt,
        train_dataset=train_set,
        val_dataset=valid_set,
        args=arg_type(**kwargs),
        training_callback=callback,
    )


PIPELINES = {
    "sft": run_sft,
    "dpo": run_dpo,
    "cpo": run_cpo,
    "orpo": run_orpo,
    "grpo": run_grpo,
    "online_dpo": run_online_family,
    "ppo": run_online_family,
    "rlhf_reinforce": run_online_family,
    "xpo": run_online_family,
}


def run(args: SimpleNamespace) -> None:
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    guard = ResourceGuard()
    studio_callback = StudioCallback(guard=guard)
    callback: TrainingCallback = studio_callback
    if getattr(args, "wandb", None):
        callback = WandBCallback(
            project_name=args.wandb,
            log_dir=args.adapter_path,
            config=vars(args),
            wrapped_callback=callback,
        )

    family = "VLM text-only" if args.model_family == "vision_language" else "Text"
    studio_log(
        f"{family} {args.train_mode.upper()} {args.train_type.upper()} run | "
        f"{args.model} | batch {args.batch_size} | lr {args.learning_rate:g} | "
        f"optimizer {args.optimizer}"
    )
    if guard_line := guard.describe():
        studio_log(guard_line)
    if args.model_family == "vision_language":
        if not args.vlm_model:
            raise ValueError("VLM mode needs an original VLM repo or local folder.")
        studio_log(f"Original VLM for final export: {args.vlm_model}")

    studio_log("Loading model")
    with quiet_vendor_output():
        model, tokenizer, adapter_file = from_pretrained(
            model=args.model,
            new_adapter_path=args.adapter_path,
            lora_config=build_lora_config(args),
            quantized_load=_quantization_config(args),
        )
        guard.release_caches()
        guard.check("loading the trainable model")

        reference_model = (
            load_reference_model(args) if args.train_mode in REFERENCE_MODES else None
        )
        guard.release_caches()
        guard.check("loading the reference model")
        judge_model, judge_tokenizer = (
            load_judge_model(args, reference_model)
            if args.train_mode in JUDGE_MODES
            else (None, None)
        )
        guard.release_caches()
        guard.check("loading the judge model")

        studio_log("Loading datasets")
        train_raw, valid_raw, test_raw = load_dataset(args, tokenizer)
        train_set = CacheDataset(train_raw)
        valid_set = CacheDataset(valid_raw)
        test_set = CacheDataset(test_raw)
        guard.release_caches()
        args.iters = _iters(args, train_raw)
        _resolve_lr_schedule(args)

        if args.resume_adapter_file:
            studio_log(f"Resuming adapter weights from {args.resume_adapter_file}")
            model.load_weights(args.resume_adapter_file, strict=False)

        print_trainable_parameters(model)
        opt = _optimizer(args)
        if args.lr_schedule:
            studio_log(f"Learning rate schedule: {args.lr_schedule}")

        guard.release_caches()
        guard.check("preparing training")
        studio_log(f"Training started ({args.iters} iterations)")
        PIPELINES[args.train_mode](
            args,
            model,
            tokenizer,
            reference_model,
            judge_model,
            judge_tokenizer,
            opt,
            train_set,
            valid_set,
            adapter_file,
            callback,
        )

        if args.test:
            test_count = _dataset_len(test_set)
            if test_count == 0:
                studio_log("Skipping test split evaluation: no test rows were found.")
            else:
                studio_log("Evaluating test split")
                test_output = TeeCapture()
                with contextlib.redirect_stdout(
                    test_output
                ), contextlib.redirect_stderr(test_output):
                    evaluate_model(
                        args=args,
                        model=model,
                        tokenizer=tokenizer,
                        reference_model=reference_model,
                        judge_model=judge_model,
                        judge_tokenizer=judge_tokenizer,
                        test_set=test_set,
                    )
                test_metrics = _test_metrics_from_output(
                    test_output.buffer.getvalue(), args.iters
                )
                if len(test_metrics) > 1:
                    studio_callback.on_test_loss_report(test_metrics)

    mx.clear_cache()
    del reference_model, judge_model, judge_tokenizer

    if args.fuse:
        if args.model_family == "vision_language":
            output_path = args.vlm_output_path or str(
                Path(args.adapter_path) / "full-vlm"
            )
            studio_log(f"Exporting full VLM to {output_path}")
            with quiet_vendor_output():
                save_pretrained_merged_vision(
                    model_name=args.vlm_model,
                    text_model=model,
                    output_path=output_path,
                    de_quantize=args.vlm_dequantize,
                )
        else:
            studio_log("Fusing merged model")
            with quiet_vendor_output():
                if args.lm_studio_name:
                    save_to_lmstudio_merged(
                        model=model,
                        tokenizer=tokenizer,
                        new_model_name=args.lm_studio_name,
                        de_quantize=args.fuse_dequantize,
                    )
                else:
                    save_pretrained_merged(
                        model=model,
                        tokenizer=tokenizer,
                        save_path=args.adapter_path,
                        de_quantize=args.fuse_dequantize,
                        remove_adapters=args.fuse_remove_adapters,
                    )
    studio_log(f"Run complete. Outputs saved under {args.adapter_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLX LoRA Studio custom training runner."
    )
    parser.add_argument(
        "--spec", required=True, help="Path to the Studio JSON run spec."
    )
    parsed = parser.parse_args()
    with open(parsed.spec, "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    try:
        run(_normalize_spec(spec))
    except Exception as exc:
        studio_log(f"Run failed: {type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
