import Foundation
import Testing
@testable import MLXLoRAStudio

@Suite("Training Config")
struct TrainingConfigTests {
    @Test("VLM settings are written to the run spec")
    func vlmSettingsAreWrittenToRunSpec() throws {
        var config = TrainingConfig()
        config.modelFamily = .visionLanguage
        config.model = "mlx-community/Qwen2.5-VL-text"
        config.vlmModel = "mlx-community/Qwen2.5-VL"
        config.vlmOutputPath = "/tmp/full-vlm"
        config.vlmDequantize = false

        let data = try config.runSpecData(adapterPath: "/tmp/adapters")
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )

        #expect(object["model_family"] as? String == "vision_language")
        #expect(object["model"] as? String == "mlx-community/Qwen2.5-VL-text")
        #expect(object["vlm_model"] as? String == "mlx-community/Qwen2.5-VL")
        #expect(object["vlm_output_path"] as? String == "/tmp/full-vlm")
        #expect(object["vlm_dequantize"] as? Bool == false)
        #expect(object["adapter_path"] as? String == "/tmp/adapters")
    }

    @Test("Text model remains the default family")
    func textModelRemainsDefaultFamily() throws {
        let config = TrainingConfig()

        let data = try config.runSpecData()
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )

        #expect(config.modelFamily == .text)
        #expect(object["model_family"] as? String == "text")
        #expect(object["vlm_model"] == nil)
        #expect(object["vlm_output_path"] == nil)
        #expect(object["vlm_dequantize"] as? Bool == true)
        #expect(object["fuse_dequantize"] as? Bool == true)
        #expect(object["fuse_remove_adapters"] as? Bool == true)
    }

    @Test("Fuse merge options are written and restored")
    func fuseMergeOptionsAreWrittenAndRestored() throws {
        var config = TrainingConfig()
        config.fuse = true
        config.fuseDequantize = false
        config.fuseRemoveAdapters = false

        let data = try config.runSpecData()
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )

        #expect(object["fuse_dequantize"] as? Bool == false)
        #expect(object["fuse_remove_adapters"] as? Bool == false)

        let restored = try TrainingConfig(jsonData: data)
        #expect(restored.fuseDequantize == false)
        #expect(restored.fuseRemoveAdapters == false)
    }

    @Test("Online user judge is written and restored")
    func onlineUserJudgeIsWrittenAndRestored() throws {
        var config = TrainingConfig()
        config.trainMode = .onlineDPO
        config.judgeKind = .user
        config.judge = "human"

        let data = try config.runSpecData()
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )

        #expect(object["judge"] as? String == "human")

        let restored = try TrainingConfig(jsonData: data)
        #expect(restored.judgeKind == .user)
        #expect(restored.judge == "human")
    }

    @Test("VLM settings round trip from saved run specs")
    func vlmSettingsRoundTripFromSavedRunSpecs() throws {
        let spec: [String: Any] = [
            "model": "text-model",
            "model_family": "vision_language",
            "vlm_model": "original-vlm",
            "vlm_output_path": "/tmp/exported-vlm",
            "vlm_dequantize": false,
            "data": "data/",
            "train_mode": "sft",
            "train_type": "lora"
        ]
        let data = try JSONSerialization.data(withJSONObject: spec)

        let config = try TrainingConfig(jsonData: data)

        #expect(config.modelFamily == .visionLanguage)
        #expect(config.model == "text-model")
        #expect(config.vlmModel == "original-vlm")
        #expect(config.vlmOutputPath == "/tmp/exported-vlm")
        #expect(config.vlmDequantize == false)
    }

    @Test("Resume candidate picks highest numbered adapter checkpoint")
    func resumeCandidatePicksHighestNumberedCheckpoint() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString, directoryHint: .isDirectory)
        let runURL = root.appending(path: "orpo-run", directoryHint: .isDirectory)
        let adaptersURL = runURL.appending(path: "adapters", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: adaptersURL, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        var config = TrainingConfig()
        config.trainMode = .orpo
        try config.runSpecData(adapterPath: adaptersURL.path)
            .write(to: runURL.appending(path: "run_spec.json"))
        for name in ["adapters.safetensors", "0000200_adapters.safetensors", "0000700_adapters.safetensors"] {
            try Data(name.utf8).write(to: adaptersURL.appending(path: name))
        }

        let run = try #require(RunArchive.discoverPersistedRuns(outputRoot: root.path).first)
        let candidate = try #require(RunArchive.resumeCandidate(for: run))

        #expect(candidate.step == 700)
        #expect(candidate.adapterFile.lastPathComponent == "0000700_adapters.safetensors")
    }

    @Test("Synthetic specs are written and discovered as previous runs")
    func syntheticSpecsAreDiscoveredAsPreviousRuns() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString, directoryHint: .isDirectory)
        let runURL = root.appending(path: "synthetic-sft", directoryHint: .isDirectory)
        let dataURL = runURL.appending(path: "generated-data", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: dataURL, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        var config = SyntheticConfig()
        config.kind = .sft
        try config.runSpecData(outputDir: dataURL.path)
            .write(to: runURL.appending(path: "synthetic_spec.json"))

        let run = try #require(RunArchive.discoverPersistedRuns(outputRoot: root.path).first)

        #expect(run.kind == .synthetic)
        #expect(run.spec == nil)
        #expect(run.title.contains("Synthetic"))
    }

    @Test("Synthetic SFT settings are written and restored")
    func syntheticSFTSettingsAreWrittenAndRestored() throws {
        var config = SyntheticConfig()
        config.kind = .sft
        config.backend = .openrouter
        config.datasetPath = "  owner/source-sft  "
        config.model = "  provider/sft-teacher  "
        config.baseURL = "  https://openrouter.ai/api/v1  "
        config.multiturn = true
        config.maxTurns = 7
        config.maxConcurrent = 6
        config.multiturnPercentile = 0.72
        config.humanRoleModel = "  provider/human-role  "
        config.runFolderName = "custom-synthetic-sft"
        config.systemPrompt = "  Be precise.  "
        config.includeSystemPrompt = true
        config.useGroundTruth = false
        config.numSamples = 123
        config.validSplit = "  validation  "
        config.testSplit = "  test  "
        config.batchSize = 3
        config.useGenerationSettings = true
        config.maxTokens = 777
        config.temperature = 0.42
        config.topP = 0.88
        config.minP = 0.03
        config.topK = 11
        config.minTokensToKeep = 2
        config.xtcProbability = 0.12
        config.xtcThreshold = 0.34
        config.seed = 99

        let data = try config.runSpecData(outputDir: "/tmp/generated-data")
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )

        #expect(object["kind"] as? String == "sft")
        #expect(object["backend"] as? String == "openrouter")
        #expect(object["dataset_path"] as? String == "owner/source-sft")
        #expect(object["model"] as? String == "provider/sft-teacher")
        #expect(object["base_url"] as? String == "https://openrouter.ai/api/v1")
        #expect(object["multiturn"] as? Bool == true)
        #expect(object["max_turns"] as? Int == 7)
        #expect(object["max_concurrent"] as? Int == 6)
        #expect(object["multiturn_percentile"] as? Double == 0.72)
        #expect(object["human_role_model"] as? String == "provider/human-role")
        #expect(object["output_dir"] as? String == "/tmp/generated-data")
        #expect(object["system_prompt"] as? String == "Be precise.")
        #expect(object["include_system_prompt"] as? Bool == true)
        #expect(object["use_ground_truth"] as? Bool == false)
        #expect(object["num_samples"] as? Int == 123)
        #expect(object["valid_split"] as? String == "validation")
        #expect(object["test_split"] as? String == "test")
        #expect(object["batch_size"] as? Int == 3)
        #expect(object["use_generation_settings"] as? Bool == true)
        #expect(object["max_tokens"] as? Int == 777)
        #expect(object["temperature"] as? Double == 0.42)
        #expect(object["top_p"] as? Double == 0.88)
        #expect(object["min_p"] as? Double == 0.03)
        #expect(object["top_k"] as? Int == 11)
        #expect(object["min_tokens_to_keep"] as? Int == 2)
        #expect(object["xtc_probability"] as? Double == 0.12)
        #expect(object["xtc_threshold"] as? Double == 0.34)
        #expect(object["seed"] as? Int == 99)

        let restored = try SyntheticConfig(jsonData: data)
        #expect(restored.kind == .sft)
        #expect(restored.backend == .openrouter)
        #expect(restored.datasetPath == "owner/source-sft")
        #expect(restored.model == "provider/sft-teacher")
        #expect(restored.baseURL == "https://openrouter.ai/api/v1")
        #expect(restored.multiturn == true)
        #expect(restored.maxTurns == 7)
        #expect(restored.maxConcurrent == 6)
        #expect(restored.multiturnPercentile == 0.72)
        #expect(restored.humanRoleModel == "provider/human-role")
        #expect(restored.systemPrompt == "Be precise.")
        #expect(restored.includeSystemPrompt == true)
        #expect(restored.useGroundTruth == false)
        #expect(restored.useGenerationSettings == true)
        #expect(restored.maxTokens == 777)
        #expect(restored.seed == 99)
    }

    @Test("Synthetic DPO settings are written and restored")
    func syntheticDPOSettingsAreWrittenAndRestored() throws {
        var config = SyntheticConfig()
        config.kind = .dpo
        config.backend = .openai
        config.datasetPath = "owner/source-dpo"
        config.baseModel = "  mlx/base-policy  "
        config.teacherModel = "  provider/dpo-teacher  "
        config.dpoGenerationTarget = .chosen
        config.baseURL = "  https://api.openai.com/v1  "
        config.resumeOutputDir = "  /tmp/previous/generated-data  "
        config.maxConcurrent = 8
        config.systemPrompt = "  Prefer concise answers.  "
        config.numSamples = 45
        config.batchSize = 4
        config.useGenerationSettings = false
        config.seed = 1234

        let data = try config.runSpecData(outputDir: "/tmp/new/generated-data")
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )

        #expect(object["kind"] as? String == "dpo")
        #expect(object["backend"] as? String == "openai")
        #expect(object["dataset_path"] as? String == "owner/source-dpo")
        #expect(object["base_model"] as? String == "mlx/base-policy")
        #expect(object["teacher_model"] as? String == "provider/dpo-teacher")
        #expect(object["dpo_generation_target"] as? String == "chosen")
        #expect(object["base_url"] as? String == "https://api.openai.com/v1")
        #expect(object["resume_output_dir"] as? String == "/tmp/previous/generated-data")
        #expect(object["output_dir"] as? String == "/tmp/new/generated-data")
        #expect(object["max_concurrent"] as? Int == 8)
        #expect(object["system_prompt"] as? String == "Prefer concise answers.")
        #expect(object["num_samples"] as? Int == 45)
        #expect(object["batch_size"] as? Int == 4)
        #expect(object["use_generation_settings"] as? Bool == false)
        #expect(object["seed"] as? Int == 1234)

        let restored = try SyntheticConfig(jsonData: data)
        #expect(restored.kind == .dpo)
        #expect(restored.backend == .openai)
        #expect(restored.baseModel == "mlx/base-policy")
        #expect(restored.teacherModel == "provider/dpo-teacher")
        #expect(restored.dpoGenerationTarget == .chosen)
        #expect(restored.resumeOutputDir == "/tmp/previous/generated-data")
        #expect(restored.outputDir == "/tmp/new/generated-data")
    }

    @Test("Metric parser handles rewards at line start")
    func metricParserHandlesRewardsAtLineStart() throws {
        let parser = MetricsParser()

        let metrics = parser.consume("Rewards 1.25, loss 2.5, iter 4")
        let metric = try #require(metrics.first)

        #expect(metric.step == 4)
        #expect(metric.values["rewards"] == 1.25)
        #expect(metric.values["loss"] == 2.5)
    }
}
