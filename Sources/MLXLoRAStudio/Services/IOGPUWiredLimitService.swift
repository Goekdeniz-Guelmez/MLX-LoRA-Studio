import Foundation

enum IOGPUWiredLimitService {
    static func apply(limitMB: Int) async throws -> String {
        let clamped = min(max(limitMB, 1_024), 1_048_576)
        return try await Task.detached(priority: .userInitiated) {
            let process = Process()
            let pipe = Pipe()
            let command = "sysctl iogpu.wired_limit_mb=\(clamped)"
            process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
            process.arguments = [
                "-e",
                "do shell script \"\(appleScriptEscaped(command))\" with administrator privileges"
            ]
            process.standardOutput = pipe
            process.standardError = pipe

            try process.run()
            process.waitUntilExit()

            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard process.terminationStatus == 0 else {
                throw IOGPUWiredLimitError.failed(output.isEmpty ? "administrator approval was cancelled or sysctl failed" : output)
            }
            return output
        }.value
    }

    private static func appleScriptEscaped(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }
}

enum IOGPUWiredLimitError: LocalizedError {
    case failed(String)

    var errorDescription: String? {
        switch self {
        case .failed(let message):
            message
        }
    }
}
