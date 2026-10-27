import Foundation
import Darwin

/// Reads the MCP backend's persisted run metadata and recent log output.
enum MCPRunDashboardReader {
    static func read(from configuredPath: String) -> [MCPTrainingRun] {
        let expanded = NSString(string: configuredPath).expandingTildeInPath
        let root = URL(fileURLWithPath: expanded, isDirectory: true)
        guard let children = try? FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }

        return children.compactMap { directory in
            let specURL = directory.appending(path: "run.json")
            guard let data = try? Data(contentsOf: specURL),
                  let record = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let id = record["id"] as? String,
                  let model = record["model"] as? String,
                  let status = record["status"] as? String else { return nil }

            let logURL = URL(fileURLWithPath: (record["log_path"] as? String) ?? directory.appending(path: "train.log").path)
            let logTail = tail(of: logURL)
            let metrics = parseMetrics(logTail)
            let savedStatus = effectiveStatus(
                status: status,
                pid: record["pid"] as? Int,
                exitCode: readExitCode(in: directory),
                stopRequested: record["stop_requested"] as? Bool ?? false
            )
            let modelInfo = metrics.last
            return MCPTrainingRun(
                id: id,
                status: savedStatus.status,
                model: model,
                trainMode: record["train_mode"] as? String ?? "training",
                trainType: record["train_type"] as? String ?? "lora",
                data: record["data"] as? String ?? "",
                iters: record["iters"] as? Int,
                currentIteration: modelInfo?.step,
                trainLoss: modelInfo?.values["loss"],
                validationLoss: modelInfo?.values["val_loss"],
                createdAt: record["created_at"] as? String ?? "",
                endedAt: record["ended_at"] as? String,
                exitCode: savedStatus.exitCode ?? record["exit_code"] as? Int,
                adapterPath: record["adapter_path"] as? String ?? "",
                logTail: logTail
            )
        }
        .sorted { $0.createdAt > $1.createdAt }
    }

    private static func parseMetrics(_ text: String) -> [TrainingMetric] {
        let parser = MetricsParser()
        return text.split(whereSeparator: \.isNewline).flatMap { parser.consume(String($0)) }
    }

    private static func readExitCode(in directory: URL) -> Int? {
        guard let text = try? String(contentsOf: directory.appending(path: "exit_code"), encoding: .utf8) else { return nil }
        return Int(text.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    private static func effectiveStatus(
        status: String,
        pid: Int?,
        exitCode: Int?,
        stopRequested: Bool
    ) -> (status: String, exitCode: Int?) {
        guard status == "running" || status == "starting" else { return (status, exitCode) }
        if let exitCode {
            return (stopRequested ? "stopped" : (exitCode == 0 ? "completed" : "failed"), exitCode)
        }
        guard let pid else { return ("failed", nil) }
        errno = 0
        if kill(pid_t(pid), 0) == 0 || errno == EPERM {
            return (status, nil)
        }
        return ("failed", nil)
    }

    private static func tail(of url: URL, maximumBytes: UInt64 = 128_000) -> String {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return "" }
        defer { try? handle.close() }
        guard let size = try? handle.seekToEnd() else { return "" }
        try? handle.seek(toOffset: size > maximumBytes ? size - maximumBytes : 0)
        guard let data = try? handle.readToEnd() else { return "" }
        return String(decoding: data, as: UTF8.self)
    }
}
