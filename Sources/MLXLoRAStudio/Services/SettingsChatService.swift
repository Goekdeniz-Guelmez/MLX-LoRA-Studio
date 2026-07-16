import Foundation

struct SettingsChatMessage: Identifiable, Equatable {
    enum Role: String { case user, assistant }
    let id = UUID()
    let role: Role
    let text: String
}

struct TrainingSettingChange: Equatable {
    let name: String
    let value: String
}

enum TrainingSettingsEditor {
    enum EditError: LocalizedError {
        case unsupported(String), invalid(String, String)
        var errorDescription: String? {
            switch self {
            case .unsupported(let name): "The assistant tried to edit unsupported setting \(name)."
            case .invalid(let name, let value): "‘\(value)’ is not valid for \(name)."
            }
        }
    }

    static let editableNames = ["model", "data", "train_mode", "train_type", "quantization", "optimizer", "batch_size", "iters", "epochs", "learning_rate", "gradient_accumulation_steps", "max_seq_length", "num_layers", "rank", "scale", "dropout", "grad_checkpoint", "steps_per_report", "steps_per_eval", "save_every", "seed"]

    @discardableResult
    static func apply(_ change: TrainingSettingChange, to config: inout TrainingConfig) throws -> String {
        let name = change.name
        let value = change.value.trimmingCharacters(in: .whitespacesAndNewlines)
        func int() throws -> Int { guard let result = Int(value) else { throw EditError.invalid(name, value) }; return result }
        func double() throws -> Double { guard let result = Double(value) else { throw EditError.invalid(name, value) }; return result }
        func bool() throws -> Bool { guard let result = Bool(value) else { throw EditError.invalid(name, value) }; return result }
        switch name {
        case "model": config.model = value
        case "data": config.data = value
        case "train_mode": guard let v = TrainMode(rawValue: value) else { throw EditError.invalid(name, value) }; config.trainMode = v
        case "train_type": guard let v = TrainType(rawValue: value) else { throw EditError.invalid(name, value) }; config.trainType = v
        case "quantization":
            let values = ["none": Quantization.none, "4bit": .fourBit, "6bit": .sixBit, "8bit": .eightBit, "mxfp4": .mxfp4]
            guard let v = values[value.lowercased()] else { throw EditError.invalid(name, value) }; config.quantization = v
        case "optimizer": guard let v = OptimizerKind(rawValue: value.lowercased()) else { throw EditError.invalid(name, value) }; config.optimizer = v
        case "batch_size": config.batchSize = max(1, try int())
        case "iters": config.iters = max(0, try int())
        case "epochs": config.epochs = max(0, try int())
        case "learning_rate": config.learningRate = try double()
        case "gradient_accumulation_steps": config.gradientAccumulationSteps = max(1, try int())
        case "max_seq_length": config.maxSeqLength = max(1, try int())
        case "num_layers": config.numLayers = max(1, try int())
        case "rank": config.rank = max(1, try int())
        case "scale": config.scale = try double()
        case "dropout": config.dropout = min(max(try double(), 0), 1)
        case "grad_checkpoint": config.gradCheckpoint = try bool()
        case "steps_per_report": config.stepsPerReport = max(1, try int())
        case "steps_per_eval": config.stepsPerEval = max(1, try int())
        case "save_every": config.saveEvery = max(1, try int())
        case "seed": config.seed = try int()
        default: throw EditError.unsupported(name)
        }
        return "\(name) = \(value)"
    }
}

enum SettingsChatService {
    struct Result { let text: String; let changes: [TrainingSettingChange] }
    enum ServiceError: LocalizedError {
        case invalidResponse, api(String)
        var errorDescription: String? { switch self { case .invalidResponse: "OpenAI returned an unreadable response."; case .api(let message): message } }
    }

    static func respond(messages: [SettingsChatMessage], config: TrainingConfig, model: String, apiKey: String) async throws -> Result {
        let history = messages.suffix(12).map { ["role": $0.role.rawValue, "content": $0.text] }
        let tool: [String: Any] = [
            "type": "function", "name": "update_training_settings", "description": "Immediately update one or more training settings when the user asks to change, tune, set, or apply them.",
            "parameters": ["type": "object", "properties": ["changes": ["type": "array", "items": ["type": "object", "properties": ["name": ["type": "string", "enum": TrainingSettingsEditor.editableNames], "value": ["type": "string"]], "required": ["name", "value"], "additionalProperties": false]]], "required": ["changes"], "additionalProperties": false], "strict": true
        ]
        let body: [String: Any] = ["model": model, "input": history, "tools": [tool], "parallel_tool_calls": false, "instructions": "You are an MLX LoRA training expert inside MLX LoRA Studio. Give concise practical advice grounded in the current configuration. Explain tradeoffs. Never claim an edit happened unless you call update_training_settings. When asked to edit, use the tool directly. Current configuration:\n\(summary(config))"]
        var request = URLRequest(url: URL(string: "https://api.openai.com/v1/responses")!)
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw ServiceError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else {
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            let error = json?["error"] as? [String: Any]
            throw ServiceError.api(error?["message"] as? String ?? "OpenAI request failed (HTTP \(http.statusCode)).")
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any], let output = json["output"] as? [[String: Any]] else { throw ServiceError.invalidResponse }
        var text = ""
        var changes: [TrainingSettingChange] = []
        for item in output {
            if item["type"] as? String == "message", let content = item["content"] as? [[String: Any]] { text += content.compactMap { $0["text"] as? String }.joined() }
            if item["type"] as? String == "function_call", item["name"] as? String == "update_training_settings", let args = item["arguments"] as? String, let argsData = args.data(using: .utf8), let object = try? JSONSerialization.jsonObject(with: argsData) as? [String: Any], let raw = object["changes"] as? [[String: Any]] {
                changes += raw.compactMap { guard let name = $0["name"] as? String, let value = $0["value"] as? String else { return nil }; return .init(name: name, value: value) }
            }
        }
        if text.isEmpty && !changes.isEmpty { text = "I updated the requested training settings." }
        guard !text.isEmpty || !changes.isEmpty else { throw ServiceError.invalidResponse }
        return Result(text: text, changes: changes)
    }

    private static func summary(_ c: TrainingConfig) -> String {
        "model=\(c.model), data=\(c.data), train_mode=\(c.trainMode.rawValue), train_type=\(c.trainType.rawValue), quantization=\(c.quantization.title), optimizer=\(c.optimizer.rawValue), batch_size=\(c.batchSize), iters=\(c.iters), epochs=\(c.epochs), learning_rate=\(c.learningRate), gradient_accumulation_steps=\(c.gradientAccumulationSteps), max_seq_length=\(c.maxSeqLength), num_layers=\(c.numLayers), rank=\(c.rank), scale=\(c.scale), dropout=\(c.dropout), grad_checkpoint=\(c.gradCheckpoint), steps_per_report=\(c.stepsPerReport), steps_per_eval=\(c.stepsPerEval), save_every=\(c.saveEvery), seed=\(c.seed)"
    }
}
