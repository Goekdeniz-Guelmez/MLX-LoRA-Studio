import Foundation

/// Coding agents supported by the one-click MCP and skill installers.
enum MCPAgentClient: String, CaseIterable, Identifiable {
    case codex
    case claudeCode
    case hermes
    case openCode

    var id: String { rawValue }

    var title: String {
        switch self {
        case .codex: "Codex"
        case .claudeCode: "Claude Code"
        case .hermes: "Hermes Agent"
        case .openCode: "OpenCode"
        }
    }

    var symbol: String {
        switch self {
        case .codex: "chevron.left.forwardslash.chevron.right"
        case .claudeCode: "sparkle"
        case .hermes: "hare"
        case .openCode: "terminal"
        }
    }

    var skillRootPath: String {
        switch self {
        case .codex: "~/.codex/skills"
        case .claudeCode: "~/.claude/skills"
        case .hermes: "~/.hermes/skills"
        case .openCode: "~/.config/opencode/skills"
        }
    }

    var executableNames: [String] {
        switch self {
        case .codex: ["codex"]
        case .claudeCode: ["claude"]
        case .hermes: ["hermes"]
        case .openCode: ["opencode"]
        }
    }
}

/// A saved run exposed by the backend MCP server.
struct MCPTrainingRun: Identifiable, Equatable {
    let id: String
    let status: String
    let model: String
    let trainMode: String
    let trainType: String
    let data: String
    let iters: Int?
    let currentIteration: Int?
    let trainLoss: Double?
    let validationLoss: Double?
    let createdAt: String
    let endedAt: String?
    let exitCode: Int?
    let adapterPath: String
    let logTail: String

    var title: String {
        let modelName = model.split(separator: "/").last.map(String.init) ?? model
        return "\(modelName) · \(trainMode.uppercased()) · \(trainType.uppercased())"
    }

    var isActive: Bool { status == "running" || status == "starting" }

    var progress: Double? {
        guard let currentIteration, let iters, iters > 0 else { return nil }
        return min(max(Double(currentIteration) / Double(iters), 0), 1)
    }

    var metricsSummary: String {
        var parts: [String] = []
        if let currentIteration {
            parts.append("Step \(currentIteration)\(iters.map { "/\($0)" } ?? "")")
        }
        if let trainLoss {
            parts.append("Loss \(trainLoss.formatted(.number.precision(.fractionLength(4))))")
        }
        if let validationLoss {
            parts.append("Val \(validationLoss.formatted(.number.precision(.fractionLength(4))))")
        }
        return parts.isEmpty ? "Waiting for metrics" : parts.joined(separator: " · ")
    }
}
