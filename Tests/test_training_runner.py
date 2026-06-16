import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TRAINERS = [
    ("cpo_trainer", "CPOTrainingArgs", "train_cpo", "cpo"),
    ("dpo_trainer", "DPOTrainingArgs", "train_dpo", "dpo"),
    ("grpo_trainer", "GRPOTrainingArgs", "train_grpo", "grpo"),
    ("online_dpo_trainer", "OnlineDPOTrainingArgs", "train_online_dpo", "online_dpo"),
    ("orpo_trainer", "ORPOTrainingArgs", "train_orpo", "orpo"),
    ("ppo_trainer", "PPOTrainingArgs", "train_ppo", "ppo"),
    (
        "rlhf_reinforce_trainer",
        "RLHFReinforceTrainingArgs",
        "train_rlhf_reinforce",
        "rlhf_reinforce",
    ),
    ("sft_trainer", "SFTTrainingArgs", "train_sft", "sft"),
    ("xpo_trainer", "XPOTrainingArgs", "train_xpo", "xpo"),
]


def training_args_factory(name):
    return type(
        name, (), {"__init__": lambda self, **kwargs: self.__dict__.update(kwargs)}
    )


def install_training_runner_stubs():
    mlx = types.ModuleType("mlx")
    mlx.core = types.ModuleType("mlx.core")
    mlx.core.random = SimpleNamespace(seed=lambda _seed: None)
    mlx.core.clear_cache = lambda: None
    mlx.core.get_peak_memory = lambda: 0
    mlx.optimizers = types.ModuleType("mlx.optimizers")

    class Optimizer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    for name in (
        "SGD",
        "RMSprop",
        "Adagrad",
        "AdaDelta",
        "Adam",
        "AdamW",
        "Adamax",
        "Lion",
        "Adafactor",
        "Muon",
    ):
        setattr(mlx.optimizers, name, Optimizer)

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.tuner = types.ModuleType("mlx_lm.tuner")
    mlx_lm.tuner.callbacks = types.ModuleType("mlx_lm.tuner.callbacks")
    mlx_lm.tuner.callbacks.TrainingCallback = object

    class WandBCallback:
        def __init__(self, **_kwargs):
            pass

    mlx_lm.tuner.callbacks.WandBCallback = WandBCallback
    mlx_lm.tuner.utils = types.ModuleType("mlx_lm.tuner.utils")
    mlx_lm.tuner.utils.build_schedule = lambda schedule: schedule
    mlx_lm.tuner.utils.load_adapters = lambda *_args, **_kwargs: None
    mlx_lm.tuner.utils.print_trainable_parameters = lambda _model: None

    package = types.ModuleType("mlx_lm_lora")
    train = types.ModuleType("mlx_lm_lora.train")
    train.CONFIG_DEFAULTS = {
        "optimizer": "adamw",
        "seed": 0,
        "batch_size": 1,
        "iters": 1,
        "epochs": None,
        "val_batches": 1,
        "steps_per_report": 1,
        "steps_per_eval": 1,
        "save_every": 1,
        "max_seq_length": 32,
        "grad_checkpoint": False,
        "gradient_accumulation_steps": 1,
        "efficient_long_context": False,
        "qat_enable": False,
        "qat_bits": 8,
        "qat_group_size": 64,
        "qat_mode": "affine",
        "qat_start_step": 1,
        "qat_interval": 1,
        "train_mode": "sft",
        "train_type": "lora",
        "model": "text-model",
        "adapter_path": "adapters",
        "data": "data/",
        "learning_rate": 1e-5,
        "load_in_4bits": False,
        "load_in_6bits": False,
        "load_in_8bits": False,
        "load_in_mxfp4": False,
        "test": False,
        "wandb": None,
        "lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0},
        "num_layers": 8,
        "lr_schedule": None,
        "fuse": False,
        "beta": 0.1,
        "reward_scaling": 1.0,
        "dpo_cpo_loss_type": "sigmoid",
        "delta": 50.0,
        "reference_model_path": None,
        "judge": "judge-model",
        "judge_system": "",
        "group_size": 4,
        "epsilon": 0.2,
        "epsilon_high": None,
        "max_completion_length": 128,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "reward_weights": None,
        "reward_functions": None,
        "reward_functions_file": None,
        "grpo_loss_type": "grpo",
        "importance_sampling_level": None,
        "alpha": [1e-5],
        "lm_studio_name": None,
    }
    train.build_lora_config = lambda _args: {"rank": 8}
    train.evaluate_model = mock.Mock()
    train.load_judge_model = mock.Mock(return_value=("judge", "judge-tokenizer"))
    train.load_reference_model = mock.Mock(return_value="reference")
    train.load_reward_functions_from_file = mock.Mock()

    datasets = types.ModuleType("mlx_lm_lora.trainer.datasets")
    datasets.CacheDataset = lambda rows: rows
    datasets.load_dataset = mock.Mock(
        return_value=([{"text": "a"}], [{"text": "b"}], [{"text": "c"}])
    )

    trainer_modules = {}
    trainer_calls = {}
    for module_name, class_name, func_name, mode in TRAINERS:
        module = types.ModuleType(f"mlx_lm_lora.trainer.{module_name}")
        setattr(module, class_name, training_args_factory(class_name))
        trainer_calls[mode] = mock.Mock(name=func_name)
        setattr(module, func_name, trainer_calls[mode])
        trainer_modules[f"mlx_lm_lora.trainer.{module_name}"] = module

    rewards = types.ModuleType("mlx_lm_lora.trainer.grpo_reward_functions")
    rewards.get_default_reward_functions = mock.Mock(return_value=["default_reward"])
    rewards.get_reward_function = mock.Mock(side_effect=lambda name: f"reward:{name}")
    rewards.list_available_reward_functions = lambda: []

    utils = types.ModuleType("mlx_lm_lora.utils")
    utils.from_pretrained = mock.Mock(
        return_value=(mock.Mock(), object(), Path("adapters/adapters.safetensors"))
    )
    utils.save_pretrained_merged = mock.Mock()
    utils.save_pretrained_merged_vision = mock.Mock()
    utils.save_to_lmstudio_merged = mock.Mock()

    modules = {
        "mlx": mlx,
        "mlx.core": mlx.core,
        "mlx.optimizers": mlx.optimizers,
        "mlx_lm": mlx_lm,
        "mlx_lm.tuner": mlx_lm.tuner,
        "mlx_lm.tuner.callbacks": mlx_lm.tuner.callbacks,
        "mlx_lm.tuner.utils": mlx_lm.tuner.utils,
        "mlx_lm_lora": package,
        "mlx_lm_lora.train": train,
        "mlx_lm_lora.trainer.datasets": datasets,
        "mlx_lm_lora.trainer.grpo_reward_functions": rewards,
        "mlx_lm_lora.utils": utils,
        **trainer_modules,
    }
    sys.modules.update(modules)
    return SimpleNamespace(
        train=train,
        datasets=datasets,
        rewards=rewards,
        trainer_calls=trainer_calls,
        utils=utils,
    )


def load_training_runner():
    stubs = install_training_runner_stubs()
    module_path = Path(__file__).resolve().parents[1] / "Backend" / "training_runner.py"
    spec = importlib.util.spec_from_file_location(
        "training_runner_for_tests", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, stubs


class TrainingRunnerTests(unittest.TestCase):
    def test_normalize_spec_defaults_to_text_family(self):
        runner, _stubs = load_training_runner()

        args = runner._normalize_spec(
            {"model": "text-model", "adapter_path": "adapters"}
        )

        self.assertEqual(args.model_family, "text")
        self.assertTrue(args.vlm_dequantize)
        self.assertTrue(args.fuse_dequantize)
        self.assertTrue(args.fuse_remove_adapters)
        self.assertIsNone(getattr(args, "vlm_model", None))
        self.assertIsNone(getattr(args, "vlm_output_path", None))

    def test_every_training_algorithm_dispatches_to_its_pipeline(self):
        expected_extra_args = {
            "sft": {},
            "dpo": {
                "beta": 0.2,
                "loss_type": "ipo",
                "delta": 7.0,
                "reference_model_path": "ref-model",
            },
            "cpo": {
                "beta": 0.2,
                "loss_type": "ipo",
                "delta": 7.0,
                "reference_model_path": "ref-model",
            },
            "orpo": {"beta": 0.2, "reward_scaling": 1.5},
            "grpo": {
                "beta": 0.2,
                "group_size": 3,
                "epsilon": 0.3,
                "epsilon_high": 0.4,
                "reference_model_path": "ref-model",
                "temperature": 0.6,
                "top_p": 0.9,
                "top_k": 12,
                "min_p": 0.05,
                "reward_weights": [0.25, 0.75],
                "importance_sampling_level": "sequence",
                "grpo_loss_type": "dr_grpo",
            },
            "online_dpo": {
                "beta": 0.2,
                "reference_model_path": "ref-model",
                "judge": "judge-model",
                "judge_system": "pick the clearer answer",
                "max_completion_length": 64,
                "loss_type": "ipo",
                "delta": 7.0,
                "temperature": 0.6,
            },
            "ppo": {
                "beta": 0.2,
                "reference_model_path": "ref-model",
                "judge": "judge-model",
                "judge_system": "pick the clearer answer",
                "max_completion_length": 64,
                "loss_type": "ipo",
                "delta": 7.0,
                "epsilon": 0.3,
                "temperature": 0.6,
            },
            "rlhf_reinforce": {
                "beta": 0.2,
                "reference_model_path": "ref-model",
                "judge": "judge-model",
                "max_completion_length": 64,
            },
            "xpo": {
                "beta": 0.2,
                "reference_model_path": "ref-model",
                "judge": "judge-model",
                "judge_system": "pick the clearer answer",
                "max_completion_length": 64,
                "loss_type": "ipo",
                "delta": 7.0,
                "temperature": 0.6,
                "alpha": [1e-5, 2e-5],
            },
        }

        for mode in [entry[3] for entry in TRAINERS]:
            with self.subTest(train_mode=mode):
                runner, stubs = load_training_runner()
                args = runner._normalize_spec(
                    {
                        "train_mode": mode,
                        "adapter_path": "adapters",
                        "fuse": False,
                        "batch_size": 2,
                        "iters": 5,
                        "val_batches": 0,
                        "steps_per_report": 2,
                        "steps_per_eval": 3,
                        "save_every": 4,
                        "max_seq_length": 256,
                        "grad_checkpoint": True,
                        "gradient_accumulation_steps": 3,
                        "efficient_long_context": True,
                        "seq_step_size": 128,
                        "qat_enable": True,
                        "qat_bits": 4,
                        "qat_group_size": 32,
                        "qat_start_step": 2,
                        "qat_interval": 5,
                        "beta": 0.2,
                        "reward_scaling": 1.5,
                        "dpo_cpo_loss_type": "ipo",
                        "delta": 7.0,
                        "reference_model_path": "ref-model",
                        "judge": "judge-model",
                        "judge_system": "pick the clearer answer",
                        "group_size": 3,
                        "epsilon": 0.3,
                        "epsilon_high": 0.4,
                        "max_completion_length": 64,
                        "temperature": 0.6,
                        "top_p": 0.9,
                        "top_k": 12,
                        "min_p": 0.05,
                        "reward_weights": "[0.25, 0.75]",
                        "reward_functions": "format, length",
                        "reward_functions_file": "/tmp/rewards.py",
                        "grpo_loss_type": "dr_grpo",
                        "importance_sampling_level": "sequence",
                        "alpha": "[0.00001, 0.00002]",
                    }
                )

                runner.run(args)

                stubs.trainer_calls[mode].assert_called_once()
                call = stubs.trainer_calls[mode].call_args.kwargs
                training_args = call["args"]
                for key, value in {
                    "batch_size": 2,
                    "iters": 5,
                    "val_batches": 0,
                    "steps_per_report": 2,
                    "steps_per_eval": 3,
                    "steps_per_save": 4,
                    "max_seq_length": 256,
                    "grad_checkpoint": True,
                    "gradient_accumulation_steps": 3,
                    "qat_enable": True,
                    "qat_bits": 4,
                    "qat_group_size": 32,
                    "qat_mode": "affine",
                    "qat_start_step": 2,
                    "qat_interval": 5,
                }.items():
                    self.assertEqual(getattr(training_args, key), value)

                if mode in {"sft", "dpo", "cpo", "orpo"}:
                    self.assertEqual(training_args.seq_step_size, 128)
                for key, value in expected_extra_args[mode].items():
                    self.assertEqual(getattr(training_args, key), value)

                if mode in runner.REFERENCE_MODES:
                    stubs.train.load_reference_model.assert_called_once()
                    self.assertEqual(call["ref_model"], "reference")
                else:
                    stubs.train.load_reference_model.assert_not_called()
                    if "ref_model" in call:
                        self.assertIsNone(call["ref_model"])

                if mode in runner.JUDGE_MODES:
                    stubs.train.load_judge_model.assert_called_once()
                    self.assertEqual(call["judge_model"], "judge")
                    self.assertEqual(call["judge_tokenizer"], "judge-tokenizer")
                else:
                    stubs.train.load_judge_model.assert_not_called()

                if mode == "grpo":
                    stubs.train.load_reward_functions_from_file.assert_called_once_with(
                        "/tmp/rewards.py"
                    )
                    self.assertEqual(
                        call["reward_funcs"], ["reward:format", "reward:length"]
                    )

    def test_vlm_mode_requires_original_vlm(self):
        runner, _stubs = load_training_runner()
        args = runner._normalize_spec(
            {
                "model_family": "vision_language",
                "model": "text-model",
                "adapter_path": "adapters",
                "fuse": True,
            }
        )

        with self.assertRaisesRegex(ValueError, "original VLM"):
            runner.run(args)

    def test_vlm_mode_exports_full_vlm(self):
        runner, stubs = load_training_runner()
        args = runner._normalize_spec(
            {
                "model_family": "vision_language",
                "model": "text-model",
                "vlm_model": "original-vlm",
                "vlm_output_path": "/tmp/full-vlm",
                "vlm_dequantize": False,
                "adapter_path": "adapters",
                "fuse": True,
            }
        )

        runner.run(args)

        stubs.utils.save_pretrained_merged_vision.assert_called_once()
        call = stubs.utils.save_pretrained_merged_vision.call_args.kwargs
        self.assertEqual(call["model_name"], "original-vlm")
        self.assertEqual(call["output_path"], "/tmp/full-vlm")
        self.assertFalse(call["de_quantize"])
        stubs.utils.save_pretrained_merged.assert_not_called()
        stubs.utils.save_to_lmstudio_merged.assert_not_called()

    def test_text_fuse_passes_merge_options(self):
        runner, stubs = load_training_runner()
        args = runner._normalize_spec(
            {
                "model_family": "text",
                "model": "text-model",
                "adapter_path": "adapters",
                "fuse": True,
                "fuse_dequantize": False,
                "fuse_remove_adapters": False,
            }
        )

        runner.run(args)

        stubs.utils.save_pretrained_merged.assert_called_once()
        call = stubs.utils.save_pretrained_merged.call_args.kwargs
        self.assertFalse(call["de_quantize"])
        self.assertFalse(call["remove_adapters"])
        stubs.utils.save_pretrained_merged_vision.assert_not_called()
        stubs.utils.save_to_lmstudio_merged.assert_not_called()

    def test_vlm_export_defaults_inside_adapter_path(self):
        runner, stubs = load_training_runner()
        args = runner._normalize_spec(
            {
                "model_family": "vision_language",
                "model": "text-model",
                "vlm_model": "original-vlm",
                "adapter_path": "/tmp/run/adapters",
                "fuse": True,
            }
        )

        runner.run(args)

        call = stubs.utils.save_pretrained_merged_vision.call_args.kwargs
        self.assertEqual(call["output_path"], "/tmp/run/adapters/full-vlm")

    def test_resource_guard_cancels_when_peak_exceeds_limit(self):
        runner, _stubs = load_training_runner()
        guard = runner.ResourceGuard()
        guard.enabled = True
        guard.max_bytes = 1_000

        with mock.patch.object(runner.mx, "get_peak_memory", return_value=2_000):
            with self.assertRaisesRegex(RuntimeError, "Resource guard cancelled"):
                guard.check("testing")

    def test_resource_guard_can_be_disabled(self):
        runner, _stubs = load_training_runner()
        guard = runner.ResourceGuard()
        guard.enabled = False
        guard.max_bytes = 1

        with mock.patch.object(runner.mx, "get_peak_memory", return_value=2_000):
            guard.check("testing")


if __name__ == "__main__":
    unittest.main()
