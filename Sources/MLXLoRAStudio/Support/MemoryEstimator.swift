import Foundation

/// Preflight memory estimator for MLX LoRA training runs.
///
/// This is an *estimate*, not a measurement. It uses a closed-form
/// approximation of how much unified memory a single forward+backward step
/// of LoRA/QLoRA fine-tuning is likely to consume on Apple silicon, and
/// compares it to the host's installed RAM to give the user a "will this
/// probably start?" signal before they hit Run.
///
/// The numbers are intentionally conservative (i.e. they err on the side
/// of "you might run out") because a hard OOM at step 50 of a 1000-step
/// run is much worse than telling someone their 7B + 4-bit will be tight.
///
/// Formula (per single training step):
///
///     peak ≈ weights
///          + optimizer_state
///          + adapter_params                (LoRA / DoRA only)
///          + activations                   (forward, scaled by grad-ckpt)
///          + reference_model                (modes that need it)
///          + judge_model                    (modes that need it, loaded on demand)
///          + framework_overhead             (MLX + Python runtime, fixed)
///          + safety_margin                  (rounded up to absorb peaks)
///
/// Weights are computed from the parameter count parsed out of the model
/// name (`Qwen3-4B` → 4B, `Llama-3.1-70B` → 70B) and the quantization
/// (`4-bit` → 0.5 bytes/param, `8-bit` → 1.0, `MXFP4` → 0.5, `None` → 2.0).
enum MemoryEstimator {

    // MARK: - Output

    struct Report: Equatable {
        /// Detected Apple-silicon chip family (e.g. "M3 Pro", "M4 Max").
        /// Falls back to "Apple silicon" if the brand string is unrecognised.
        let chipLabel: String
        /// Total unified-memory size in bytes.
        let totalMemory: UInt64
        /// Estimated peak memory the proposed training run will use, in bytes.
        let estimatedPeak: UInt64
        /// "Likely fits" / "Risky" / "Too large" verdict.
        let verdict: Verdict
        /// Human-readable line of breakdown numbers (params, batch, seq, etc.).
        let summary: String
        /// Optional actionable suggestion if verdict is not green.
        let suggestion: String?

        var usedRatio: Double {
            guard totalMemory > 0 else { return 0 }
            return Double(estimatedPeak) / Double(totalMemory)
        }
    }

    enum Verdict: String, Equatable {
        case likelyFits = "Likely fits"
        case risky = "Risky"
        case tooLarge = "Too large"

        var symbol: String {
            switch self {
            case .likelyFits: "checkmark.circle.fill"
            case .risky: "exclamationmark.triangle.fill"
            case .tooLarge: "xmark.octagon.fill"
            }
        }
    }

    // MARK: - Public entry point

    /// Build an estimate for the given training config on the current host.
    static func estimate(for config: TrainingConfig) -> Report {
        let (chipLabel, totalMemory) = detectHost()
        let params = parseParamCount(from: config.model)
        let bytesPerParam = quantizationBytes(config.quantization)
        let hiddenSize = estimateHiddenSize(params: params)
        let numLayers = max(config.numLayers, defaultLayerCount(params: params))

        let weights = params.map { UInt64(Double($0) * bytesPerParam) } ?? 0

        // Optimizer state: Adam-family keeps two moments per trainable param
        // (in fp32 by default in MLX); others keep just one or none.
        let optimizerMultiplier = optimizerStateMultiplier(config.optimizer)

        // Adapter params are tiny relative to weights but they DO live in
        // memory. We approximate by counting target modules per layer.
        let adapterParams: UInt64 = {
            guard config.trainType != .full else { return 0 }
            // Each target linear is rank * (in + out) parameters; with the
            // MLX defaults we hit q/k/v/o + gate/up/down, ~7 linears.
            let linearsPerLayer = 7
            let rank = max(config.rank, 1)
            let perLinear = UInt64(rank) * UInt64(hiddenSize) * 2
            return UInt64(numLayers) * UInt64(linearsPerLayer) * perLinear * 2 // A+B
        }()

        let optimizerState: UInt64 = {
            guard let params else { return 0 }
            let trainable: UInt64
            if config.trainType == .full {
                trainable = UInt64(params)
            } else {
                trainable = adapterParams
            }
            return trainable * 4 * UInt64(optimizerMultiplier) // fp32
        }()

        // Activations scale with batch × seq × hidden × layers. With grad
        // checkpointing we only keep one layer's activations at a time.
        let batch = max(config.batchSize * config.gradientAccumulationSteps, 1)
        let seq = max(config.maxSeqLength, 1)
        let activations: UInt64 = {
            let perLayer = UInt64(batch) * UInt64(seq) * UInt64(hiddenSize) * 2 // bytes
            let visibleLayers = config.gradCheckpoint ? 1 : UInt64(numLayers)
            return perLayer * visibleLayers * 4 // a few cached intermediates
        }()

        // Reference model (DPO/RL modes need a frozen copy of the base model)
        // and judge model (online DPO/XPO/PPO/RLHF load a second LM for
        // scoring). They sit idle most of the time but do take RAM.
        let referenceModel: UInt64 = config.trainMode.needsReference ? weights : 0
        let judgeModel: UInt64 = config.trainMode.needsJudge ? UInt64(Double(params ?? 0) * bytesPerParam) : 0

        // MLX + Python + tokenizer caches, etc. Empirical floor: 1.5 GB.
        let frameworkOverhead: UInt64 = 1_500_000_000

        // 15% safety margin so we don't promise what we can't deliver at
        // the very first step (peak spikes during attention are common).
        let rawTotal = weights
            + optimizerState
            + adapterParams
            + activations
            + referenceModel
            + judgeModel
            + frameworkOverhead
        let safetyMultiplier: Double = 1.15
        let estimatedPeak = UInt64(Double(rawTotal) * safetyMultiplier)

        let verdict = classify(estimated: estimatedPeak, total: totalMemory)
        let summary = buildSummary(
            config: config,
            params: params,
            bytesPerParam: bytesPerParam,
            batch: batch,
            seq: seq,
            numLayers: numLayers,
            hiddenSize: hiddenSize,
            weights: weights
        )
        let suggestion = verdict == .likelyFits ? nil : suggest(
            for: config,
            verdict: verdict,
            params: params,
            bytesPerParam: bytesPerParam
        )

        return Report(
            chipLabel: chipLabel,
            totalMemory: totalMemory,
            estimatedPeak: estimatedPeak,
            verdict: verdict,
            summary: summary,
            suggestion: suggestion
        )
    }

    // MARK: - Host detection

    private static func detectHost() -> (label: String, memory: UInt64) {
        let memory = sysctlU64(name: "hw.memsize") ?? 0
        let brand = sysctlString(name: "machdep.cpu.brand_string") ?? "Apple silicon"
        let label = humaniseChip(brand: brand)
        return (label, memory)
    }

    private static func humaniseChip(brand: String) -> String {
        // Brand strings look like "Apple M3 Pro" / "Apple M4 Max" / "Apple M1".
        // We trim "Apple " and pass the rest through, with a sensible
        // fallback for virtualised / older hosts.
        let trimmed = brand
            .replacingOccurrences(of: "Apple ", with: "")
            .trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { return "Apple silicon" }
        // If the brand string is e.g. "VirtualApple" we still want to keep it.
        return trimmed
    }

    // MARK: - Config parsing

    /// Parse `Qwen3-4B` / `Llama-3.1-70B-Instruct` / `phi-3-mini-3.8B` into
    /// a parameter count. Returns nil if we can't find a B-suffix — the UI
    /// then shows "Unknown" rather than guessing.
    static func parseParamCount(from model: String) -> Int? {
        // Strip the org prefix (everything up to and including the first /)
        let bare = model.split(separator: "/").last.map(String.init) ?? model

        // Look for the first "B" or "M" suffix preceded by a number. We do
        // a regex because model names mix dashes/dots/underscores freely.
        // Examples we want to match:
        //   "Qwen3-4B"               → 4
        //   "Llama-3.1-70B-Instruct" → 70
        //   "phi-3-mini-3.8B"        → 3.8
        //   "gemma-2-9b"             → 9 (lowercase tolerated)
        let pattern = #"(\d+(?:\.\d+)?)\s*([BM])\b"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
            return nil
        }
        let range = NSRange(bare.startIndex..<bare.endIndex, in: bare)
        guard let match = regex.firstMatch(in: bare, options: [], range: range),
              match.numberOfRanges >= 3,
              let numberRange = Range(match.range(at: 1), in: bare),
              let unitRange = Range(match.range(at: 2), in: bare) else {
            return nil
        }
        let numberString = bare[numberRange]
        let unit = bare[unitRange].uppercased()
        guard let baseValue = Double(numberString) else { return nil }
        let params: Double
        switch unit {
        case "B": params = baseValue * 1_000_000_000
        case "M": params = baseValue * 1_000_000
        default: return nil
        }
        return Int(params.rounded())
    }

    static func quantizationBytes(_ q: Quantization) -> Double {
        switch q {
        case .none: return 2.0   // fp16/bf16
        case .eightBit: return 1.0
        case .sixBit: return 0.75
        case .fourBit: return 0.5
        case .mxfp4: return 0.5
        }
    }

    static func optimizerStateMultiplier(_ kind: OptimizerKind) -> Int {
        // Adam/AdamW/Adamax/Lion keep two fp32 moments; AdaFactor keeps
        // one factored pair; SGD/RMSprop/Adagrad/Adadelta keep one
        // (momentum or squared-grad) buffer; Muon keeps a momentum buffer.
        switch kind {
        case .adam, .adamw, .adamax, .lion: return 2
        case .adafactor: return 1
        case .muon, .rmsprop, .adagrad, .adadelta: return 1
        case .sgd: return 1
        }
    }

    /// Best-guess hidden size for a model with the given param count. This
    /// is intentionally rough — without loading config.json we can only
    /// estimate. We use a power-law fit that lands near the published
    /// values for Qwen / Llama / Mistral family sizes we ship defaults for.
    static func estimateHiddenSize(params: Int?) -> Int {
        guard let params else { return 2048 }
        switch params {
        case ..<1_000_000_000: return 1024       // < 1B
        case ..<3_000_000_000: return 1536       // 1–3B
        case ..<8_000_000_000: return 2560       // 3–8B (Qwen3-4B ≈ 2560)
        case ..<15_000_000_000: return 4096      // 8–15B
        case ..<35_000_000_000: return 5120      // 15–35B
        case ..<80_000_000_000: return 8192      // 35–80B
        default: return 12288                     // ≥ 80B
        }
    }

    /// Best-guess layer count for a model with the given param count. If
    /// the user has explicitly set `numLayers` in the config we use that
    /// instead (caller's responsibility), but the default of 16 in
    /// `TrainingConfig` is wrong for 70B+ models and we want the estimate
    /// to reflect reality.
    static func defaultLayerCount(params: Int?) -> Int {
        guard let params else { return 16 }
        switch params {
        case ..<1_000_000_000: return 12
        case ..<3_000_000_000: return 24
        case ..<8_000_000_000: return 36
        case ..<15_000_000_000: return 40
        case ..<35_000_000_000: return 48
        case ..<80_000_000_000: return 80
        default: return 96
        }
    }

    // MARK: - Verdict + suggestion

    private static func classify(estimated: UInt64, total: UInt64) -> Verdict {
        guard total > 0 else { return .risky }
        let ratio = Double(estimated) / Double(total)
        if ratio < 0.70 { return .likelyFits }
        if ratio < 0.90 { return .risky }
        return .tooLarge
    }

    private static func suggest(
        for config: TrainingConfig,
        verdict: Verdict,
        params: Int?,
        bytesPerParam: Double
    ) -> String {
        // Build a short list of "what to change" hints ordered by impact.
        // The most effective single change is almost always to reduce
        // quant precision (e.g. 8-bit → 4-bit halves weights) or batch
        // size (shrinks activations and KV cache quadratically with seq).
        var hints: [String] = []

        if verdict == .tooLarge, config.quantization != .fourBit, config.quantization != .mxfp4 {
            hints.append("Try 4-bit quantization")
        } else if verdict == .risky, config.quantization == .none {
            hints.append("Try 4-bit quantization")
        }

        let effectiveBatch = config.batchSize * config.gradientAccumulationSteps
        if effectiveBatch > 1 {
            hints.append("Lower batch size")
        }

        if config.maxSeqLength >= 4096 {
            hints.append("Reduce max sequence length")
        }

        if !config.gradCheckpoint, params.map({ $0 >= 8_000_000_000 }) ?? false {
            hints.append("Enable grad checkpointing")
        }

        if config.trainType == .full {
            hints.append("Switch to LoRA")
        }

        if config.trainMode.needsReference {
            hints.append("Use a smaller reference model")
        }

        if hints.isEmpty {
            // Fall back to a generic "free RAM" hint so we always say
            // something useful.
            hints.append("Close other apps to free RAM")
        }

        return hints.joined(separator: " · ")
    }

    private static func buildSummary(
        config: TrainingConfig,
        params: Int?,
        bytesPerParam: Double,
        batch: Int,
        seq: Int,
        numLayers: Int,
        hiddenSize: Int,
        weights: UInt64
    ) -> String {
        let paramsLabel: String
        if let params {
            if params >= 1_000_000_000 {
                paramsLabel = String(format: "%.1fB params", Double(params) / 1_000_000_000)
            } else {
                paramsLabel = String(format: "%.0fM params", Double(params) / 1_000_000)
            }
        } else {
            paramsLabel = "params?"
        }
        let quantLabel: String = {
            switch config.quantization {
            case .none: "fp16"
            case .eightBit: "8-bit"
            case .sixBit: "6-bit"
            case .fourBit: "4-bit"
            case .mxfp4: "MXFP4"
            }
        }()
        let trainTypeLabel: String = {
            switch config.trainType {
            case .lora: "LoRA"
            case .dora: "DoRA"
            case .full: "Full FT"
            }
        }()
        return "\(paramsLabel) · \(quantLabel) · \(trainTypeLabel) · batch \(batch) · seq \(seq) · \(numLayers)L × \(hiddenSize)h"
    }

    // MARK: - sysctl helpers

    private static func sysctlString(name: String) -> String? {
        var size: size_t = 0
        sysctlbyname(name, nil, &size, nil, 0)
        guard size > 0 else { return nil }
        var buffer = [UInt8](repeating: 0, count: size)
        let result = sysctlbyname(name, &buffer, &size, nil, 0)
        guard result == 0 else { return nil }
        // Truncate at the first NUL byte (sysctl strings are C-strings) and
        // decode as UTF-8 — the new constructor handles the NUL terminator
        // without flagging a deprecation warning.
        let bytes = buffer.prefix { $0 != 0 }
        return String(decoding: bytes, as: UTF8.self)
    }

    private static func sysctlU64(name: String) -> UInt64? {
        var value: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        let result = sysctlbyname(name, &value, &size, nil, 0)
        guard result == 0 else { return nil }
        return value
    }
}

// MARK: - Byte formatting

extension MemoryEstimator.Report {
    /// Human-readable "X.X GB" string for the estimated peak.
    var estimatedPeakString: String {
        ByteCountFormatter.string(fromByteCount: Int64(estimatedPeak), countStyle: .memory)
    }

    /// Human-readable "X.X GB" string for the total host memory.
    var totalMemoryString: String {
        ByteCountFormatter.string(fromByteCount: Int64(totalMemory), countStyle: .memory)
    }
}
