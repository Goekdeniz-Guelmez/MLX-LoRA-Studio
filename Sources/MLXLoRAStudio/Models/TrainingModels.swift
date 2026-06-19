import Foundation
import Observation

enum SidebarSection: String, CaseIterable, Identifiable {
    case train
    case metrics
    case synthetic
    case upload
    case guide
    case runs
    case about

    var id: String { rawValue }

    var title: String {
        switch self {
        case .train: "Train"
        case .metrics: "Live Metrics"
        case .synthetic: "Synthetic Data"
        case .upload: "Upload to HF"
        case .guide: "Algorithm Guide"
        case .runs: "Runs"
        case .about: "About"
        }
    }

    var symbol: String {
        switch self {
        case .train: "cpu"
        case .metrics: "chart.line.uptrend.xyaxis"
        case .synthetic: "sparkles"
        case .upload: "arrow.up.circle"
        case .guide: "book"
        case .runs: "chart.xyaxis.line"
        case .about: "info.circle"
        }
    }
}

enum TrainMode: String, CaseIterable, Identifiable {
    case sft
    case dpo
    case cpo
    case orpo
    case grpo
    case onlineDPO = "online_dpo"
    case xpo
    case rlhfReinforce = "rlhf_reinforce"
    case ppo

    var id: String { rawValue }

    var title: String {
        switch self {
        case .sft: "SFT"
        case .dpo: "DPO"
        case .cpo: "CPO"
        case .orpo: "ORPO"
        case .grpo: "GRPO"
        case .onlineDPO: "Online DPO"
        case .xpo: "XPO"
        case .rlhfReinforce: "RLHF Reinforce"
        case .ppo: "PPO"
        }
    }

    var family: String {
        switch self {
        case .sft: "Supervised"
        case .dpo, .cpo, .orpo: "Preference"
        case .grpo, .onlineDPO, .xpo, .rlhfReinforce, .ppo: "RL / Online"
        }
    }

    var summary: String {
        switch self {
        case .sft:
            "Learns from prompt/completion or chat examples. Best first step for style, domain, and instruction following."
        case .dpo:
            "Optimizes chosen responses over rejected responses using a frozen reference model."
        case .cpo:
            "Preference optimization without a separate reference model path in the main loop."
        case .orpo:
            "Combines supervised learning and preference odds into one efficient preference objective."
        case .grpo:
            "Generates groups of completions and improves behavior with reward functions."
        case .onlineDPO:
            "Samples completions during training and uses a judge to create online preferences."
        case .xpo:
            "Online preference optimization variant with an alpha control for update strength."
        case .rlhfReinforce:
            "Policy-gradient style RLHF loop using judge rewards."
        case .ppo:
            "Classic clipped policy optimization with reference and judge feedback."
        }
    }

    var datasetHint: String {
        switch self {
        case .sft: "Default: mlx-community/JOSIE-v2-Instruct-5K"
        case .dpo, .cpo: "Default: mlx-community/Human-Like-DPO"
        case .orpo: "Default: mlx-community/Josiefied-Qwen3-dpo-v1-flat"
        case .grpo: "Default: mlx-community/Dolci-Think-RL-7B-2k"
        case .onlineDPO, .xpo, .rlhfReinforce, .ppo: "Default: mlx-community/Human-Like-DPO"
        }
    }

    var defaultDataset: String {
        switch self {
        case .sft:
            "mlx-community/JOSIE-v2-Instruct-5K"
        case .dpo, .cpo:
            "mlx-community/Human-Like-DPO"
        case .orpo:
            "mlx-community/Josiefied-Qwen3-dpo-v1-flat"
        case .grpo:
            "mlx-community/Dolci-Think-RL-7B-2k"
        case .onlineDPO, .xpo, .rlhfReinforce, .ppo:
            "mlx-community/Human-Like-DPO"
        }
    }

    var needsReference: Bool {
        switch self {
        case .dpo, .grpo, .onlineDPO, .xpo, .rlhfReinforce, .ppo: true
        case .sft, .cpo, .orpo: false
        }
    }

    var needsJudge: Bool {
        switch self {
        case .onlineDPO, .xpo, .rlhfReinforce, .ppo: true
        case .sft, .dpo, .cpo, .orpo, .grpo: false
        }
    }

    var supportsQAT: Bool {
        switch self {
        case .sft, .dpo, .orpo: true
        default: false
        }
    }
}

enum JudgeKind: String, CaseIterable, Identifiable {
    case llm
    case user

    var id: String { rawValue }

    var title: String {
        switch self {
        case .llm: "LLM"
        case .user: "User"
        }
    }
}

enum TrainType: String, CaseIterable, Identifiable {
    case lora
    case dora
    case full

    var id: String { rawValue }
    var title: String { rawValue.uppercased() }
}

enum ModelFamily: String, CaseIterable, Identifiable {
    case text
    case visionLanguage = "vision_language"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .text: "Text LLM"
        case .visionLanguage: "Vision-Language"
        }
    }
}

enum Quantization: String, CaseIterable, Identifiable {
    case none
    case fourBit
    case sixBit
    case eightBit
    case mxfp4

    var id: String { rawValue }

    var title: String {
        switch self {
        case .none: "None"
        case .fourBit: "4-bit"
        case .sixBit: "6-bit"
        case .eightBit: "8-bit"
        case .mxfp4: "MXFP4"
        }
    }
}

enum OptimizerKind: String, CaseIterable, Identifiable {
    case sgd
    case rmsprop
    case adagrad
    case adadelta
    case adam
    case adamw
    case adamax
    case lion
    case adafactor
    case muon

    var id: String { rawValue }

    var title: String {
        switch self {
        case .sgd: "SGD"
        case .rmsprop: "RMSprop"
        case .adagrad: "Adagrad"
        case .adadelta: "AdaDelta"
        case .adam: "Adam"
        case .adamw: "AdamW"
        case .adamax: "Adamax"
        case .lion: "Lion"
        case .adafactor: "Adafactor"
        case .muon: "Muon"
        }
    }
}

enum LearningRateScheduleKind: String, CaseIterable, Identifiable {
    case constant
    case cosineDecay = "cosine_decay"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .constant: "Constant"
        case .cosineDecay: "Cosine Decay"
        }
    }
}

struct GRPORewardFunction: Identifiable, Equatable {
    let name: String
    let title: String
    let summary: String

    var id: String { name }

    static let defaults: [GRPORewardFunction] = [
        GRPORewardFunction(
            name: "r1_accuracy_reward_func",
            title: "Accuracy",
            summary: "2.0 when the extracted <answer> exactly matches the dataset answer."
        ),
        GRPORewardFunction(
            name: "r1_int_reward_func",
            title: "Integer Answer",
            summary: "0.5 when the extracted <answer> is a digit-only integer."
        ),
        GRPORewardFunction(
            name: "r1_strict_format_reward_func",
            title: "Strict Format",
            summary: "0.5 for strict <think> ... </think> <answer> ... </answer> output."
        ),
        GRPORewardFunction(
            name: "r1_soft_format_reward_func",
            title: "Soft Format",
            summary: "0.5 when think and answer XML tags appear in the right order with content."
        ),
        GRPORewardFunction(
            name: "r1_count_xml",
            title: "XML Count",
            summary: "Small score for exactly one set of think/answer tags, with trailing text penalty."
        )
    ]
}

enum SyntheticKind: String, CaseIterable, Identifiable {
    case sft
    case dpo

    var id: String { rawValue }
    var title: String { rawValue.uppercased() }

    var defaultDataset: String {
        "mlx-community/ultrafeedback-prompts-flat-rlhf"
    }
}

enum SyntheticDPOGenerationTarget: String, CaseIterable, Identifiable {
    case both
    case chosen
    case rejected

    var id: String { rawValue }

    var title: String {
        switch self {
        case .both: "Chosen + rejected"
        case .chosen: "Chosen only"
        case .rejected: "Rejected only"
        }
    }

    var summary: String {
        switch self {
        case .both:
            "Generate a full DPO triple: teacher response as chosen, base response as rejected."
        case .chosen:
            "Generate or regenerate only chosen and keep rejected from the source dataset."
        case .rejected:
            "Generate or regenerate only rejected and keep chosen from the source dataset."
        }
    }
}

enum SyntheticBackend: String, CaseIterable, Identifiable {
    case mlx
    case openai
    case openrouter
    case ollama
    case lmstudio
    case omlx
    case custom

    var id: String { rawValue }

    var title: String {
        switch self {
        case .mlx: "MLX local"
        case .openai: "OpenAI"
        case .openrouter: "OpenRouter"
        case .ollama: "Ollama"
        case .lmstudio: "LM Studio"
        case .omlx: "oMLX"
        case .custom: "Custom"
        }
    }

    var defaultBaseURL: String {
        switch self {
        case .mlx: ""
        case .openai: "https://api.openai.com/v1"
        case .openrouter: "https://openrouter.ai/api/v1"
        case .ollama: "http://localhost:11434/v1"
        case .lmstudio: "http://localhost:1234/v1"
        case .omlx: "http://localhost:8060/v1"
        case .custom: ""
        }
    }
}

enum HFModelUploadKind: String, CaseIterable, Identifiable {
    case adaptersOnly
    case mergedWeights

    var id: String { rawValue }

    var title: String {
        switch self {
        case .adaptersOnly: "Adapters Only"
        case .mergedWeights: "Merged Weights"
        }
    }
}

enum HFUploadTarget: String, CaseIterable, Identifiable {
    case model
    case dataset
    case all

    var id: String { rawValue }

    var title: String {
        switch self {
        case .model: "Model"
        case .dataset: "Synthetic Dataset"
        case .all: "Model + Dataset"
        }
    }
}

struct HFUploadConfig: Equatable {
    var modelUploadKind: HFModelUploadKind = .adaptersOnly
    var localModelPath = ""
    var modelRepo = ""
    var uploadSyntheticDataset = false
    var localDatasetPath = ""
    var datasetRepo = ""
    var privateRepo = false
    var commitMessage = "Upload from MLX LoRA Studio"

    func runSpecData(target: HFUploadTarget) throws -> Data {
        var spec: [String: Any] = [
            "upload_target": target.rawValue,
            "private": privateRepo,
            "commit_message": commitMessage
        ]
        if target == .model || target == .all {
            spec["model_upload_kind"] = modelUploadKind.rawValue
            spec["local_model_path"] = localModelPath
            spec["model_repo"] = modelRepo
        }
        if target == .dataset || target == .all {
            spec["upload_synthetic_dataset"] = target == .dataset ? true : uploadSyntheticDataset
            spec["local_dataset_path"] = localDatasetPath
            spec["dataset_repo"] = datasetRepo
        }
        spec = spec.mapValues { value in
            if let string = value as? String {
                return string.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            return value
        }
        return try JSONSerialization.data(withJSONObject: spec, options: [.prettyPrinted, .sortedKeys])
    }
}

struct TrainingConfig: Equatable {
    static let defaultModel = "Qwen/Qwen3-0.6B"
    static let builtInDatasets = Set(TrainMode.allCases.map(\.defaultDataset) + ["data/"])

    var model = Self.defaultModel
    var modelFamily: ModelFamily = .text
    var vlmModel = ""
    var vlmOutputPath = ""
    var vlmDequantize = true
    var data = TrainMode.sft.defaultDataset
    var adapterPath = "adapters"
    var runFolderName = ""
    var lmStudioName = ""
    var trainMode: TrainMode = .sft
    var trainType: TrainType = .lora
    var quantization: Quantization = .fourBit
    var optimizer: OptimizerKind = .adamw
    var seed = 0
    var numLayers = 16
    var batchSize = 1
    var iters = 1000
    var epochs = 0
    var gradientAccumulationSteps = 1
    var valBatches = 25
    var learningRate = 0.00001
    var learningRateSchedule: LearningRateScheduleKind = .constant
    var lrWarmupSteps = 40
    var lrWarmupInit = 0.0000001
    var lrDecayFraction = 0.8
    var lrFinal = 0.000002
    var stepsPerReport = 10
    var stepsPerEval = 200
    var saveEvery = 100
    var test = false
    var testBatches = 100
    var maxSeqLength = 2048
    var gradCheckpoint = true
    var efficientLongContext = false
    var seqStepSize = 512
    var maskPrompt = false
    var fuse = true
    var fuseDequantize = true
    var fuseRemoveAdapters = true
    var resumeAdapterFile = ""
    var rank = 8
    var scale = 20.0
    var dropout = 0.0
    var beta = 0.1
    var rewardScaling = 1.0
    var dpoCpoLossType = "sigmoid"
    var delta = 50.0
    var referenceModelPath = ""
    var judgeKind: JudgeKind = .llm
    var judge = "Qwen/Qwen3-0.6B"
    var groupSize = 4
    var epsilon = 0.0001
    var epsilonHigh = ""
    var maxCompletionLength = 512
    var temperature = 0.8
    var topP = 0.95
    var topK = 20
    var minP = 0.0
    var rewardWeights = ""
    var rewardFunctions = ""
    var rewardFunctionsFile = ""
    var grpoLossType = "grpo"
    var importanceSamplingLevel = ""
    var judgeSystem = ""
    var alpha = "0.00001"

    var promptFeature = ""
    var completionFeature = ""
    var chosenFeature = ""
    var rejectedFeature = ""
    var chatFeature = ""
    var textFeature = ""
    var systemFeature = ""
    var preferenceScoreFeature = ""
    var answerFeature = ""
    var typeFeature = ""
    var qatEnable = false
    var qatBits = 8
    var qatGroupSize = 64
    var qatStartStep = 1
    var qatInterval = 1

    var selectedGRPORewardFunctionNames: [String] {
        let names = rewardFunctions
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !names.isEmpty else {
            return GRPORewardFunction.defaults.map(\.name)
        }
        return names
    }

    mutating func toggleDefaultGRPORewardFunction(_ name: String) {
        var selected = selectedGRPORewardFunctionNames
        if selected.contains(name) {
            selected.removeAll { $0 == name }
        } else {
            selected.append(name)
        }
        rewardFunctions = selected.joined(separator: ", ")
    }

    mutating func useDefaultGRPORewardFunctions() {
        rewardFunctions = ""
    }

    mutating func applyDefaultDataset(for mode: TrainMode) {
        guard Self.builtInDatasets.contains(data) else { return }
        data = mode.defaultDataset
    }

    func automaticRunFolderName(date: Date = .now) -> String {
        let modelName = model
            .split(separator: "/")
            .last
            .map(String.init) ?? "model"
        return RunFolderNamer.makeName(
            pieces: [modelName, trainType.rawValue, trainMode.rawValue],
            date: date
        )
    }

    func resolvedRunFolderName(date: Date = .now) -> String {
        let customName = runFolderName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !customName.isEmpty else {
            return automaticRunFolderName(date: date)
        }
        return RunFolderNamer.sanitize(customName)
    }

    func runSpecData(adapterPath overrideAdapterPath: String? = nil) throws -> Data {
        var spec: [String: Any] = [
            "model": model,
            "model_family": modelFamily.rawValue,
            "train": true,
            "train_type": trainType.rawValue,
            "train_mode": trainMode.rawValue,
            "optimizer": optimizer.rawValue,
            "data": data,
            "seed": seed,
            "num_layers": numLayers,
            "batch_size": batchSize,
            "gradient_accumulation_steps": gradientAccumulationSteps,
            "val_batches": valBatches,
            "learning_rate": learningRate,
            "steps_per_report": stepsPerReport,
            "steps_per_eval": stepsPerEval,
            "adapter_path": overrideAdapterPath ?? adapterPath,
            "save_every": saveEvery,
            "test": test,
            "test_batches": testBatches,
            "max_seq_length": maxSeqLength,
            "grad_checkpoint": gradCheckpoint,
            "efficient_long_context": efficientLongContext,
            "seq_step_size": seqStepSize,
            "mask_prompt": maskPrompt,
            "fuse": fuse,
            "fuse_dequantize": fuseDequantize,
            "fuse_remove_adapters": fuseRemoveAdapters,
            "beta": beta,
            "reward_scaling": rewardScaling,
            "dpo_cpo_loss_type": dpoCpoLossType,
            "delta": delta,
            "group_size": groupSize,
            "epsilon": epsilon,
            "max_completion_length": maxCompletionLength,
            "temperature": temperature,
            "top_p": topP,
            "top_k": topK,
            "min_p": minP,
            "grpo_loss_type": grpoLossType,
            "lora_parameters": [
                "rank": rank,
                "scale": scale,
                "dropout": dropout
            ],
            "load_in_4bits": quantization == .fourBit,
            "load_in_6bits": quantization == .sixBit,
            "load_in_8bits": quantization == .eightBit,
            "load_in_mxfp4": quantization == .mxfp4,
            "qat_enable": qatEnable,
            "qat_bits": qatBits,
            "qat_group_size": qatGroupSize,
            "qat_mode": "affine",
            "qat_start_step": qatStartStep,
            "qat_interval": qatInterval
        ]

        if epochs > 0 {
            spec["epochs"] = epochs
            spec["iters"] = NSNull()
        } else {
            spec["iters"] = iters
            spec["epochs"] = NSNull()
        }
        if let lrSchedule = learningRateScheduleSpec {
            spec["lr_schedule"] = lrSchedule
        }
        appendSpecString(&spec, "lm_studio_name", lmStudioName)
        appendSpecString(&spec, "vlm_model", vlmModel)
        appendSpecString(&spec, "vlm_output_path", vlmOutputPath)
        spec["vlm_dequantize"] = vlmDequantize
        appendSpecString(&spec, "reference_model_path", referenceModelPath)
        if trainMode.needsJudge { appendSpecString(&spec, "judge", judge) }
        appendSpecString(&spec, "epsilon_high", epsilonHigh)
        appendSpecString(&spec, "reward_weights", rewardWeights)
        appendSpecString(&spec, "reward_functions", rewardFunctions)
        appendSpecString(&spec, "reward_functions_file", rewardFunctionsFile)
        appendSpecString(&spec, "importance_sampling_level", importanceSamplingLevel)
        appendSpecString(&spec, "judge_system", judgeSystem)
        appendSpecString(&spec, "alpha", alpha)
        appendSpecString(&spec, "prompt_feature", promptFeature)
        appendSpecString(&spec, "completion_feature", completionFeature)
        appendSpecString(&spec, "chosen_feature", chosenFeature)
        appendSpecString(&spec, "rejected_feature", rejectedFeature)
        appendSpecString(&spec, "chat_feature", chatFeature)
        appendSpecString(&spec, "text_feature", textFeature)
        appendSpecString(&spec, "system_feature", systemFeature)
        appendSpecString(&spec, "preference_score_feature", preferenceScoreFeature)
        appendSpecString(&spec, "answer_feature", answerFeature)
        appendSpecString(&spec, "type_feature", typeFeature)
        appendSpecString(&spec, "resume_adapter_file", resumeAdapterFile)

        return try JSONSerialization.data(
            withJSONObject: spec,
            options: [.prettyPrinted, .sortedKeys]
        )
    }

    private var learningRateScheduleSpec: [String: Any]? {
        switch learningRateSchedule {
        case .constant:
            nil
        case .cosineDecay:
            [
                "name": LearningRateScheduleKind.cosineDecay.rawValue,
                "warmup": max(0, lrWarmupSteps),
                "warmup_init": lrWarmupInit,
                "arguments": [
                    learningRate,
                    ["iters_fraction": min(max(lrDecayFraction, 0.0), 1.0)],
                    lrFinal
                ]
            ]
        }
    }
}

struct SyntheticConfig: Equatable {
    static let builtInDatasets = Set(SyntheticKind.allCases.map(\.defaultDataset))

    var kind: SyntheticKind = .sft
    var backend: SyntheticBackend = .mlx
    var datasetPath = SyntheticKind.sft.defaultDataset
    var model = "Goekdeniz-Guelmez/JOSIE-1.1-4B-Instruct"
    var baseURL = ""
    var apiKey = ""
    var multiturn = false
    var maxTurns = 4
    var maxConcurrent = 4
    var multiturnPercentile = 0.6
    var humanRoleModel = ""
    var baseModel = "Qwen/Qwen3-0.6B"
    var teacherModel = "Goekdeniz-Guelmez/JOSIE-1.1-4B-Instruct"
    var dpoGenerationTarget: SyntheticDPOGenerationTarget = .both
    var outputDir = "./output"
    var resumeOutputDir = ""
    var runFolderName = ""
    var systemPrompt = ""
    var includeSystemPrompt = false
    var useGroundTruth = true
    var numSamples = 10000
    var validSplit = ""
    var testSplit = ""
    var batchSize = 2
    var useGenerationSettings = false
    var maxTokens = 4096
    var temperature = 0.6
    var topP = 0.95
    var minP = 0.0
    var topK = 20
    var minTokensToKeep = 1
    var xtcProbability = 0.0
    var xtcThreshold = 0.0
    var seed = 42

    mutating func applyDefaultDataset(for kind: SyntheticKind) {
        guard Self.builtInDatasets.contains(datasetPath) else { return }
        datasetPath = kind.defaultDataset
    }

    mutating func applyBackendDefaults(for backend: SyntheticBackend) {
        if baseURL.isEmpty || SyntheticBackend.allCases.map(\.defaultBaseURL).contains(baseURL) {
            baseURL = backend.defaultBaseURL
        }

        let defaultProviderModel = "Goekdeniz-Guelmez/JOSIE-1.1-4B-Instruct"
        if backend == .mlx {
            if model.isEmpty {
                model = defaultProviderModel
            }
            if teacherModel.isEmpty {
                teacherModel = defaultProviderModel
            }
        }
    }

    func automaticRunFolderName(date: Date = .now) -> String {
        RunFolderNamer.makeName(
            pieces: ["synthetic", kind.rawValue, datasetPath.split(separator: "/").last.map(String.init) ?? "dataset"],
            date: date
        )
    }

    func resolvedRunFolderName(date: Date = .now) -> String {
        let customName = runFolderName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !customName.isEmpty else {
            return automaticRunFolderName(date: date)
        }
        return RunFolderNamer.sanitize(customName)
    }

    func runSpecData(outputDir overrideOutputDir: String? = nil) throws -> Data {
        var spec: [String: Any] = [
            "kind": kind.rawValue,
            "backend": backend.rawValue,
            "dataset_path": datasetPath,
            "model": model,
            "base_url": baseURL,
            "multiturn": multiturn,
            "max_turns": maxTurns,
            "max_concurrent": maxConcurrent,
            "multiturn_percentile": multiturnPercentile,
            "human_role_model": humanRoleModel,
            "base_model": baseModel,
            "teacher_model": teacherModel,
            "dpo_generation_target": dpoGenerationTarget.rawValue,
            "output_dir": overrideOutputDir ?? outputDir,
            "resume_output_dir": resumeOutputDir,
            "system_prompt": systemPrompt,
            "include_system_prompt": includeSystemPrompt,
            "use_ground_truth": useGroundTruth,
            "num_samples": numSamples,
            "valid_split": validSplit,
            "test_split": testSplit,
            "batch_size": batchSize,
            "use_generation_settings": useGenerationSettings,
            "max_tokens": maxTokens,
            "temperature": temperature,
            "top_p": topP,
            "min_p": minP,
            "top_k": topK,
            "min_tokens_to_keep": minTokensToKeep,
            "xtc_probability": xtcProbability,
            "xtc_threshold": xtcThreshold,
            "seed": seed
        ]
        spec = spec.mapValues { value in
            if let string = value as? String {
                return string.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            return value
        }
        return try JSONSerialization.data(withJSONObject: spec, options: [.prettyPrinted, .sortedKeys])
    }
}

extension SyntheticConfig {
    init(jsonData data: Data) throws {
        self.init()
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        Self.applySpec(object, to: &self)
    }

    static func decoded(from specURL: URL) -> SyntheticConfig? {
        guard let data = try? Data(contentsOf: specURL) else { return nil }
        return try? Self(jsonData: data)
    }

    private static func applySpec(_ spec: [String: Any], to config: inout SyntheticConfig) {
        if let v = spec["kind"] as? String, let kind = SyntheticKind(rawValue: v) { config.kind = kind }
        if let v = spec["backend"] as? String, let backend = SyntheticBackend(rawValue: v) { config.backend = backend }
        if let v = spec["dataset_path"] as? String { config.datasetPath = v }
        if let v = spec["model"] as? String { config.model = v }
        if let v = spec["base_url"] as? String { config.baseURL = v }
        if let v = spec["human_role_model"] as? String { config.humanRoleModel = v }
        if let v = spec["base_model"] as? String { config.baseModel = v }
        if let v = spec["teacher_model"] as? String { config.teacherModel = v }
        if let v = spec["dpo_generation_target"] as? String,
           let target = SyntheticDPOGenerationTarget(rawValue: v) {
            config.dpoGenerationTarget = target
        }
        if let v = spec["output_dir"] as? String { config.outputDir = v }
        if let v = spec["resume_output_dir"] as? String { config.resumeOutputDir = v }
        if let v = spec["system_prompt"] as? String { config.systemPrompt = v }
        config.multiturn = boolValue(spec["multiturn"]) ?? config.multiturn
        config.includeSystemPrompt = boolValue(spec["include_system_prompt"]) ?? config.includeSystemPrompt
        config.useGroundTruth = boolValue(spec["use_ground_truth"]) ?? config.useGroundTruth
        config.useGenerationSettings = boolValue(spec["use_generation_settings"]) ?? config.useGenerationSettings
        config.maxTurns = intValue(spec["max_turns"]) ?? config.maxTurns
        config.maxConcurrent = intValue(spec["max_concurrent"]) ?? config.maxConcurrent
        config.numSamples = intValue(spec["num_samples"]) ?? config.numSamples
        config.batchSize = intValue(spec["batch_size"]) ?? config.batchSize
        config.maxTokens = intValue(spec["max_tokens"]) ?? config.maxTokens
        config.topK = intValue(spec["top_k"]) ?? config.topK
        config.minTokensToKeep = intValue(spec["min_tokens_to_keep"]) ?? config.minTokensToKeep
        config.seed = intValue(spec["seed"]) ?? config.seed
        config.multiturnPercentile = doubleValue(spec["multiturn_percentile"]) ?? config.multiturnPercentile
        config.temperature = doubleValue(spec["temperature"]) ?? config.temperature
        config.topP = doubleValue(spec["top_p"]) ?? config.topP
        config.minP = doubleValue(spec["min_p"]) ?? config.minP
        config.xtcProbability = doubleValue(spec["xtc_probability"]) ?? config.xtcProbability
        config.xtcThreshold = doubleValue(spec["xtc_threshold"]) ?? config.xtcThreshold
        if let v = spec["valid_split"] as? String { config.validSplit = v }
        if let v = spec["test_split"] as? String { config.testSplit = v }
    }

    private static func intValue(_ value: Any?) -> Int? {
        switch value {
        case let v as Int: return v
        case let v as NSNumber: return v.intValue
        case let v as Double: return Int(v)
        case let v as String: return Int(v)
        case is NSNull: return nil
        default: return nil
        }
    }

    private static func doubleValue(_ value: Any?) -> Double? {
        switch value {
        case let v as Double: return v
        case let v as NSNumber: return v.doubleValue
        case let v as Int: return Double(v)
        case let v as String: return Double(v)
        case is NSNull: return nil
        default: return nil
        }
    }

    private static func boolValue(_ value: Any?) -> Bool? {
        switch value {
        case let v as Bool: return v
        case let v as NSNumber: return v.boolValue
        case is NSNull: return nil
        default: return nil
        }
    }
}

struct TrainingMetric: Identifiable, Equatable, Codable {
    var id: UUID = UUID()
    let step: Int
    let values: [String: Double]
    let rawLine: String

    init(step: Int, values: [String: Double], rawLine: String) {
        self.step = step
        self.values = values
        self.rawLine = rawLine
    }

    // Convenience accessors used by code that only needs the headline
    // SFT-style metrics. Live Metrics reads from `values` directly.
    var loss: Double? { values["loss"] }
    var validationLoss: Double? { values["val_loss"] }
    var learningRate: Double? { values["learning_rate"] ?? values["lr"] }
    var memoryGB: Double? { values["peak_mem"] ?? values["memory"] }
}

// MARK: - Persisted training metrics (on disk in the run folder)
enum TrainingMetricIO {
    static let filename = "metrics.json"

    static func write(_ metrics: [TrainingMetric], to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(metrics)
        try data.write(to: url, options: .atomic)
    }

    static func read(from url: URL) -> [TrainingMetric] {
        guard let data = try? Data(contentsOf: url) else { return [] }
        let decoder = JSONDecoder()
        guard let decoded = try? decoder.decode([TrainingMetric].self, from: data) else {
            return []
        }
        return decoded.sorted { $0.step < $1.step }
    }
}

// MARK: - Previous runs (discovered from the output root)
struct PersistedRun: Identifiable, Equatable {
    enum Kind: String, Equatable {
        case training
        case synthetic
        case hfUpload
    }

    let id: String
    let folderURL: URL
    let kind: Kind
    let createdAt: Date
    let title: String
    let spec: TrainingConfig?
    let metrics: [TrainingMetric]
    let syntheticSpec: SyntheticConfig?
    let syntheticSamples: [SyntheticDatasetSample]
    let command: String

    var hasMetrics: Bool { !metrics.isEmpty }
    var hasSpec: Bool { spec != nil }
    var hasSyntheticSpec: Bool { syntheticSpec != nil }

    var finalLoss: Double? { metrics.reversed().compactMap(\.loss).first }
    var finalValidationLoss: Double? { metrics.reversed().compactMap(\.validationLoss).first }
}

struct SyntheticDatasetSample: Identifiable, Equatable {
    let index: Int
    let sourceFile: String
    let fields: [SyntheticSampleField]

    var id: String { "\(sourceFile)#\(index)" }
}

struct SyntheticSampleField: Identifiable, Equatable {
    let key: String
    let value: String

    var id: String { key }
}

struct TrainingResumeCandidate: Equatable {
    let adapterFile: URL
    let step: Int?
}

struct RunRecord: Identifiable, Equatable {
    let id: UUID
    let title: String
    let command: String
    let startedAt: Date
    var endedAt: Date?
    var status: String

    init(id: UUID = UUID(), title: String, command: String, startedAt: Date, endedAt: Date? = nil, status: String) {
        self.id = id
        self.title = title
        self.command = command
        self.startedAt = startedAt
        self.endedAt = endedAt
        self.status = status
    }
}

struct LocalRunOutput: Identifiable, Equatable {
    let name: String
    let path: String

    var id: String { path }
}

private func appendSpecString(_ spec: inout [String: Any], _ key: String, _ value: String) {
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return }
    spec[key] = trimmed
}

enum RunFolderNamer {
    static func makeName(pieces: [String], date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd-HHmmss"
        let prefix = pieces
            .map(sanitize)
            .filter { !$0.isEmpty }
            .joined(separator: "-")
        return "\(prefix)-\(formatter.string(from: date))"
    }

    static func sanitize(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
        let scalars = value.unicodeScalars.map { scalar in
            allowed.contains(scalar) ? Character(scalar) : "-"
        }
        let collapsed = String(scalars)
            .split(separator: "-", omittingEmptySubsequences: true)
            .joined(separator: "-")
            .trimmingCharacters(in: CharacterSet(charactersIn: "-_. "))
        return collapsed.isEmpty ? "run" : collapsed
    }
}

// MARK: - TrainingConfig JSON decoder

extension TrainingConfig {
    init(jsonData: Data) throws {
        guard let object = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            throw NSError(
                domain: "TrainingConfig",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "run_spec.json is not a JSON object"]
            )
        }
        self.init()
        Self.applySpec(object, to: &self)
    }

    static func decoded(from specURL: URL) -> TrainingConfig? {
        guard let data = try? Data(contentsOf: specURL) else { return nil }
        return try? Self(jsonData: data)
    }

    var reconstructedCommand: String {
        let adapter = adapterPath.isEmpty ? "adapters" : adapterPath
        var parts = ["python -m mlx_lm_lora train"]
        parts += ["--model", model]
        parts += ["--train-mode", trainMode.rawValue]
        parts += ["--train-type", trainType.rawValue]
        parts += ["--data", data]
        parts += ["--adapter-path", adapter]
        if !referenceModelPath.isEmpty {
            parts += ["--reference-model-path", referenceModelPath]
        }
        if trainMode.needsJudge, !judge.isEmpty {
            parts += ["--judge", judge]
        }
        return parts
            .map { $0.contains(" ") ? "\"\($0)\"" : $0 }
            .joined(separator: " ")
    }

    // MARK: - Spec application

    private static func applySpec(_ spec: [String: Any], to config: inout TrainingConfig) {
        if let v = spec["model"] as? String { config.model = v }
        if let v = spec["model_family"] as? String, let family = ModelFamily(rawValue: v) {
            config.modelFamily = family
        }
        if let v = spec["vlm_model"] as? String { config.vlmModel = v }
        if let v = spec["vlm_output_path"] as? String { config.vlmOutputPath = v }
        if let v = spec["data"] as? String { config.data = v }
        if let v = spec["adapter_path"] as? String { config.adapterPath = v }
        if let v = spec["train_mode"] as? String, let mode = TrainMode(rawValue: v) {
            config.trainMode = mode
        }
        if let v = spec["train_type"] as? String, let type = TrainType(rawValue: v) {
            config.trainType = type
        }
        if let v = spec["optimizer"] as? String, let opt = OptimizerKind(rawValue: v) {
            config.optimizer = opt
        }

        // Numeric scalars
        config.seed = intValue(spec["seed"]) ?? config.seed
        config.numLayers = intValue(spec["num_layers"]) ?? config.numLayers
        config.batchSize = intValue(spec["batch_size"]) ?? config.batchSize
        config.iters = intValue(spec["iters"]) ?? config.iters
        config.epochs = intValue(spec["epochs"]) ?? config.epochs
        config.gradientAccumulationSteps = intValue(spec["gradient_accumulation_steps"])
            ?? config.gradientAccumulationSteps
        config.valBatches = intValue(spec["val_batches"]) ?? config.valBatches
        config.learningRate = doubleValue(spec["learning_rate"]) ?? config.learningRate
        config.stepsPerReport = intValue(spec["steps_per_report"]) ?? config.stepsPerReport
        config.stepsPerEval = intValue(spec["steps_per_eval"]) ?? config.stepsPerEval
        config.saveEvery = intValue(spec["save_every"]) ?? config.saveEvery
        config.testBatches = intValue(spec["test_batches"]) ?? config.testBatches
        config.maxSeqLength = intValue(spec["max_seq_length"]) ?? config.maxSeqLength
        config.seqStepSize = intValue(spec["seq_step_size"]) ?? config.seqStepSize
        config.lrWarmupSteps = intValue(spec["warmup"]) ?? config.lrWarmupSteps
        config.qatBits = intValue(spec["qat_bits"]) ?? config.qatBits
        config.qatGroupSize = intValue(spec["qat_group_size"]) ?? config.qatGroupSize
        config.qatStartStep = intValue(spec["qat_start_step"]) ?? config.qatStartStep
        config.qatInterval = intValue(spec["qat_interval"]) ?? config.qatInterval

        // Booleans
        config.test = boolValue(spec["test"]) ?? config.test
        config.gradCheckpoint = boolValue(spec["grad_checkpoint"]) ?? config.gradCheckpoint
        config.efficientLongContext = boolValue(spec["efficient_long_context"])
            ?? config.efficientLongContext
        config.maskPrompt = boolValue(spec["mask_prompt"]) ?? config.maskPrompt
        config.fuse = boolValue(spec["fuse"]) ?? config.fuse
        config.fuseDequantize = boolValue(spec["fuse_dequantize"]) ?? config.fuseDequantize
        config.fuseRemoveAdapters = boolValue(spec["fuse_remove_adapters"]) ?? config.fuseRemoveAdapters
        config.qatEnable = boolValue(spec["qat_enable"]) ?? config.qatEnable
        config.vlmDequantize = boolValue(spec["vlm_dequantize"]) ?? config.vlmDequantize

        // Floats
        config.beta = doubleValue(spec["beta"]) ?? config.beta
        config.rewardScaling = doubleValue(spec["reward_scaling"]) ?? config.rewardScaling
        config.delta = doubleValue(spec["delta"]) ?? config.delta
        config.epsilon = doubleValue(spec["epsilon"]) ?? config.epsilon
        config.maxCompletionLength = intValue(spec["max_completion_length"])
            ?? config.maxCompletionLength
        config.temperature = doubleValue(spec["temperature"]) ?? config.temperature
        config.topP = doubleValue(spec["top_p"]) ?? config.topP
        config.topK = intValue(spec["top_k"]) ?? config.topK
        config.minP = doubleValue(spec["min_p"]) ?? config.minP
        config.lrWarmupInit = doubleValue(spec["warmup_init"]) ?? config.lrWarmupInit
        config.lrDecayFraction = doubleValue(spec["iters_fraction"]) ?? config.lrDecayFraction
        config.lrFinal = doubleValue(spec["lr_final"]) ?? config.lrFinal

        // Strings (and stringified numerics that the writer always emits as strings)
        if let v = spec["dpo_cpo_loss_type"] as? String { config.dpoCpoLossType = v }
        if let v = spec["grpo_loss_type"] as? String { config.grpoLossType = v }
        if let v = spec["epsilon_high"] as? String { config.epsilonHigh = v }
        if let v = spec["reward_weights"] as? String { config.rewardWeights = v }
        if let v = spec["reward_functions"] as? String { config.rewardFunctions = v }
        if let v = spec["reward_functions_file"] as? String { config.rewardFunctionsFile = v }
        if let v = spec["importance_sampling_level"] as? String { config.importanceSamplingLevel = v }
        if let v = spec["judge_system"] as? String { config.judgeSystem = v }
        if let v = spec["alpha"] as? String { config.alpha = v }
        if let v = spec["judge"] as? String {
            config.judge = v
            config.judgeKind = v.trimmingCharacters(in: .whitespacesAndNewlines)
                .localizedCaseInsensitiveCompare("human") == .orderedSame ? .user : .llm
        }
        if let v = spec["reference_model_path"] as? String { config.referenceModelPath = v }
        if let v = spec["resume_adapter_file"] as? String { config.resumeAdapterFile = v }
        if let v = spec["lm_studio_name"] as? String { config.lmStudioName = v }
        if let v = spec["prompt_feature"] as? String { config.promptFeature = v }
        if let v = spec["completion_feature"] as? String { config.completionFeature = v }
        if let v = spec["chosen_feature"] as? String { config.chosenFeature = v }
        if let v = spec["rejected_feature"] as? String { config.rejectedFeature = v }
        if let v = spec["chat_feature"] as? String { config.chatFeature = v }
        if let v = spec["text_feature"] as? String { config.textFeature = v }
        if let v = spec["system_feature"] as? String { config.systemFeature = v }
        if let v = spec["preference_score_feature"] as? String { config.preferenceScoreFeature = v }
        if let v = spec["answer_feature"] as? String { config.answerFeature = v }
        if let v = spec["type_feature"] as? String { config.typeFeature = v }

        // Quantization is encoded as one of four booleans in the spec.
        if let v = spec["load_in_4bits"] as? Bool, v { config.quantization = .fourBit }
        else if let v = spec["load_in_6bits"] as? Bool, v { config.quantization = .sixBit }
        else if let v = spec["load_in_8bits"] as? Bool, v { config.quantization = .eightBit }
        else if let v = spec["load_in_mxfp4"] as? Bool, v { config.quantization = .mxfp4 }

        // LoRA parameters (rank/scale/dropout) live in a sub-dict;
        if let lora = spec["lora_parameters"] as? [String: Any] {
            config.rank = intValue(lora["rank"]) ?? config.rank
            config.scale = doubleValue(lora["scale"]) ?? config.scale
            config.dropout = doubleValue(lora["dropout"]) ?? config.dropout
        }
        config.rank = intValue(spec["rank"]) ?? config.rank
        config.scale = doubleValue(spec["scale"]) ?? config.scale
        config.dropout = doubleValue(spec["dropout"]) ?? config.dropout

        if let schedule = spec["lr_schedule"] as? [String: Any],
           let name = schedule["name"] as? String,
           let kind = LearningRateScheduleKind(rawValue: name) {
            config.learningRateSchedule = kind
            if let warmup = intValue(schedule["warmup"]) {
                config.lrWarmupSteps = warmup
            }
            if let warmupInit = doubleValue(schedule["warmup_init"]) {
                config.lrWarmupInit = warmupInit
            }
            if let args = schedule["arguments"] as? [Any], args.count >= 3 {
                if let lr = doubleValue(args[0]) { config.learningRate = lr }
                if let fractionDict = args[1] as? [String: Any],
                   let fraction = doubleValue(fractionDict["iters_fraction"]) {
                    config.lrDecayFraction = fraction
                }
                if let lrFinal = doubleValue(args[2]) { config.lrFinal = lrFinal }
            }
        }
    }

    private static func intValue(_ value: Any?) -> Int? {
        switch value {
        case let v as Int: return v
        case let v as NSNumber: return v.intValue
        case let v as Double: return Int(v)
        case let v as String: return Int(v)
        case is NSNull: return nil
        default: return nil
        }
    }

    private static func doubleValue(_ value: Any?) -> Double? {
        switch value {
        case let v as Double: return v
        case let v as NSNumber: return v.doubleValue
        case let v as Int: return Double(v)
        case let v as String: return Double(v)
        case is NSNull: return nil
        default: return nil
        }
    }

    private static func boolValue(_ value: Any?) -> Bool? {
        switch value {
        case let v as Bool: return v
        case let v as NSNumber: return v.boolValue
        case is NSNull: return nil
        default: return nil
        }
    }
}
