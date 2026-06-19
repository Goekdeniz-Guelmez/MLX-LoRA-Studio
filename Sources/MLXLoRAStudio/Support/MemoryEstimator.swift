import Foundation

enum MemoryEstimator {

    // MARK: - Output
    struct Report: Equatable {
        let chipLabel: String
        let totalMemory: UInt64
        let estimatedPeak: UInt64
        let verdict: Verdict
        let summary: String
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
    static func estimate(for config: TrainingConfig) -> Report {
        let (chipLabel, totalMemory) = detectHost()
        let params = parseParamCount(from: config.model)
        let bytesPerParam = quantizationBytes(config.quantization)
        let hiddenSize = estimateHiddenSize(params: params)
        let numLayers = max(config.numLayers, defaultLayerCount(params: params))

        let weights = params.map { UInt64(Double($0) * bytesPerParam) } ?? 0

        let optimizerMultiplier = optimizerStateMultiplier(config.optimizer)

        let adapterParams: UInt64 = {
            guard config.trainType != .full else { return 0 }
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

        let batch = max(config.batchSize * config.gradientAccumulationSteps, 1)
        let seq = max(config.maxSeqLength, 1)
        let activations: UInt64 = {
            let perLayer = UInt64(batch) * UInt64(seq) * UInt64(hiddenSize) * 2 // bytes
            let visibleLayers = config.gradCheckpoint ? 1 : UInt64(numLayers)
            return perLayer * visibleLayers * 4 // a few cached intermediates
        }()

        let referenceModel: UInt64 = config.trainMode.needsReference ? weights : 0
        let judgeModel: UInt64 = config.trainMode.needsJudge ? UInt64(Double(params ?? 0) * bytesPerParam) : 0

        let frameworkOverhead: UInt64 = 1_500_000_000

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
        let trimmed = brand
            .replacingOccurrences(of: "Apple ", with: "")
            .trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { return "Apple silicon" }
        return trimmed
    }

    // MARK: - Config parsing
    static func parseParamCount(from model: String) -> Int? {
        let bare = model.split(separator: "/").last.map(String.init) ?? model

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
        case .none: return 2.0
        case .eightBit: return 1.0
        case .sixBit: return 0.75
        case .fourBit: return 0.5
        case .mxfp4: return 0.5
        }
    }

    static func optimizerStateMultiplier(_ kind: OptimizerKind) -> Int {
        switch kind {
        case .adam, .adamw, .adamax, .lion: return 2
        case .adafactor: return 1
        case .muon, .rmsprop, .adagrad, .adadelta: return 1
        case .sgd: return 1
        }
    }

    static func estimateHiddenSize(params: Int?) -> Int {
        guard let params else { return 2048 }
        switch params {
        case ..<1_000_000_000: return 1024
        case ..<3_000_000_000: return 1536
        case ..<8_000_000_000: return 2560
        case ..<15_000_000_000: return 4096
        case ..<35_000_000_000: return 5120
        case ..<80_000_000_000: return 8192
        default: return 12288
        }
    }

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
    var estimatedPeakString: String {
        ByteCountFormatter.string(fromByteCount: Int64(estimatedPeak), countStyle: .memory)
    }

    var totalMemoryString: String {
        ByteCountFormatter.string(fromByteCount: Int64(totalMemory), countStyle: .memory)
    }
}
