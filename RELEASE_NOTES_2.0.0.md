# MLX LoRA Studio 2.0.0

This release updates MLX LoRA Studio for `mlx-lm-lora` 3.0.0 and brings its new training workflows into the native Mac interface.

## Highlights

- Added **Final Token Preference Optimization (FTPO)** with Hugging Face and local Antidoom dataset support.
- Added all FTPO objective controls: target MSE lambda/tau, non-target MSE lambda, and logit clipping epsilon.
- Added selectable SFT objectives: standard **NLL**, memory-bounded **Chunked NLL**, and **Dynamic Fine-Tuning (DFT)**.
- Pinned the bundled Python environment to `mlx-lm-lora==3.0.0`.
- Removed synthetic dataset creation from the active navigation and onboarding flow, matching the upstream 3.0.0 removal.
- Preserved read-only discovery of existing synthetic runs so prior outputs remain accessible.
- Expanded the in-app Algorithm Guide and wiki with dedicated FTPO and Dynamic Fine-Tuning documentation.
- Updated saved-run serialization, the Runs inspector, backend dispatch tests, and native configuration tests for the new settings.

## FTPO datasets

FTPO accepts either a Hugging Face dataset repository or local JSONL data. Rows must contain:

- `context_with_chat_template`
- `rejected_decoded`
- `multi_chosen_decoded`

## Validation

- Native Swift build passed.
- 11 Swift configuration and archive tests passed.
- 8 Python training-runner tests passed across every supported training mode.
- Python syntax compilation and patch whitespace validation passed.

## Installation

Download the DMG, drag **MLX LoRA Studio** into `/Applications`, and launch it normally. If macOS blocks an ad-hoc signed build, follow the first-launch instructions in the README.
