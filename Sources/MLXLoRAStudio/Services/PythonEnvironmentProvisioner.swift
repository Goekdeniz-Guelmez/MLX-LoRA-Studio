import Foundation
@preconcurrency import Dispatch

enum PythonProvisionerError: LocalizedError {
    case commandFailed(exitCode: Int32, stderr: String)
    case condaNotFound
    case noBasePython

    var errorDescription: String? {
        switch self {
        case .commandFailed(let exitCode, let stderr):
            let trimmed = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            return "Provisioning failed (exit \(exitCode)).\n\(trimmed)"
        case .condaNotFound:
            return "Could not find a `conda` binary on PATH or in common install locations."
        case .noBasePython:
            return "Could not determine the base Python version to seed the new env with."
        }
    }
}

@MainActor
enum PythonEnvironmentProvisioner {

    /// Default parent directory used for new venvs: `~/MLXLoRAStudio/envs`.
    static func defaultVenvParent() -> URL {
        URL(fileURLWithPath: NSHomeDirectory())
            .appending(path: "MLXLoRAStudio", directoryHint: .isDirectory)
            .appending(path: "envs", directoryHint: .isDirectory)
    }

    static func createVenv(
        name: String,
        basePython: String,
        parentDir: URL,
        log: (@MainActor @Sendable (String) -> Void)? = nil
    ) async throws -> URL {
        try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)
        let target = parentDir.appending(path: name, directoryHint: .isDirectory)
        if FileManager.default.fileExists(atPath: target.path) {
            throw PythonProvisionerError.commandFailed(
                exitCode: -1,
                stderr: "An environment already exists at \(target.path)."
            )
        }

        log?("Creating venv at \(target.path) using \(basePython)…")
        let result = await runCapturing(
            executable: basePython,
            arguments: ["-m", "venv", target.path],
            log: log
        )
        try throwOnFailure(result, log: log)

        let python = target.appending(path: "bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            throw PythonProvisionerError.commandFailed(
                exitCode: -1,
                stderr: "Venv was created but no executable was found at \(python.path)."
            )
        }
        log?("Venv ready: \(python.path)")
        return python
    }

    static func createCondaEnv(
        name: String,
        basePython: String,
        condaBin: String? = nil,
        log: (@MainActor @Sendable (String) -> Void)? = nil
    ) async throws -> URL {
        let conda = try condaBin ?? findConda()
        let condaRoot = URL(fileURLWithPath: conda).deletingLastPathComponent().deletingLastPathComponent()
        let version = try await pythonVersionString(of: basePython)
        guard let majorMinor = Self.majorMinor(from: version) else {
            throw PythonProvisionerError.noBasePython
        }

        log?("Creating conda env \(name) (python=\(majorMinor)) using \(conda)…")
        let result = await runCapturing(
            executable: conda,
            arguments: ["create", "-n", name, "python=\(majorMinor)", "-y"],
            log: log
        )
        try throwOnFailure(result, log: log)

        let python = condaRoot
            .appending(path: "envs", directoryHint: .isDirectory)
            .appending(path: name, directoryHint: .isDirectory)
            .appending(path: "bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            throw PythonProvisionerError.commandFailed(
                exitCode: -1,
                stderr: "Conda env was created but no executable was found at \(python.path)."
            )
        }
        log?("Conda env ready: \(python.path)")
        return python
    }

    // MARK: - Helpers

    private static func findConda() throws -> String {
        let candidates = [
            "\(NSHomeDirectory())/miniconda3/bin/conda",
            "\(NSHomeDirectory())/anaconda3/bin/conda",
            "\(NSHomeDirectory())/miniforge3/bin/conda",
            "\(NSHomeDirectory())/mambaforge/bin/conda",
            "\(NSHomeDirectory())/opt/miniconda3/bin/conda",
            "\(NSHomeDirectory())/opt/anaconda3/bin/conda",
            "/opt/homebrew/bin/conda",
            "/usr/local/bin/conda",
            "/opt/anaconda3/bin/conda",
            "/opt/miniconda3/bin/conda"
        ]
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            return path
        }
        throw PythonProvisionerError.condaNotFound
    }

    private static func pythonVersionString(of executable: String) async throws -> String {
        let result = await runCapturing(
            executable: executable,
            arguments: ["-c", "import platform; print(platform.python_version())"]
        )
        guard result.status == 0 else { throw PythonProvisionerError.noBasePython }
        return result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func majorMinor(from version: String) -> String? {
        let parts = version.split(separator: ".")
        guard parts.count >= 2 else { return nil }
        return "\(parts[0]).\(parts[1])"
    }

    private struct CapturedResult {
        let status: Int32
        let stdout: String
        let stderr: String
    }

    private static func throwOnFailure(_ result: CapturedResult, log: ((String) -> Void)?) throws {
        guard result.status != 0 else { return }
        log?("Command failed with exit \(result.status).")
        throw PythonProvisionerError.commandFailed(exitCode: result.status, stderr: result.stderr)
    }

    private static func runCapturing(
        executable: String,
        arguments: [String],
        log: (@MainActor @Sendable (String) -> Void)? = nil
    ) async -> CapturedResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe

        do {
            try process.run()
        } catch {
            return CapturedResult(status: -1, stdout: "", stderr: "")
        }

        let buffers = PipeBuffers()
        let group = DispatchGroup()
        group.enter()
        group.enter()
        let outHandle = outPipe.fileHandleForReading
        let errHandle = errPipe.fileHandleForReading
        DispatchQueue.global(qos: .userInitiated).async {
            buffers.stdout = outHandle.readDataToEndOfFile()
            group.leave()
        }
        DispatchQueue.global(qos: .userInitiated).async {
            buffers.stderr = errHandle.readDataToEndOfFile()
            group.leave()
        }

        return await withCheckedContinuation { continuation in
            process.terminationHandler = { proc in
                group.notify(queue: .global()) {
                    let stdout = String(data: buffers.stdout, encoding: .utf8) ?? ""
                    let stderr = String(data: buffers.stderr, encoding: .utf8) ?? ""
                    let lines = Self.collectLogLines(stdout: stdout, stderr: stderr)
                    Task { @MainActor in
                        if let log {
                            for line in lines { log(line) }
                        }
                        continuation.resume(returning: CapturedResult(status: proc.terminationStatus, stdout: stdout, stderr: stderr))
                    }
                }
            }
        }
    }

    private nonisolated static func collectLogLines(stdout: String, stderr: String) -> [String] {
        var result: [String] = []
        for raw in stdout.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(raw)
            if !line.isEmpty { result.append(line) }
        }
        for raw in stderr.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(raw)
            if !line.isEmpty { result.append(line) }
        }
        return result
    }
}

private final class PipeBuffers: @unchecked Sendable {
    var stdout = Data()
    var stderr = Data()
}
