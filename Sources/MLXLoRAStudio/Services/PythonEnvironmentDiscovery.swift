import Foundation
@preconcurrency import Dispatch

/// Scans the host system for available Python interpreters — system binaries,
/// Homebrew, pyenv, uv, conda envs, and project venvs — and returns a deduplicated
/// list of `PythonEnvironment` values ready to show in the Settings picker.
///
/// The scan is best-effort: any failed probe is silently skipped, timeouts are
/// bounded, and the function always returns within a few seconds even on a
/// messy machine.
enum PythonEnvironmentDiscovery {

    /// Scans for every Python interpreter it can find.
    static func scan() async -> [PythonEnvironment] {
        let candidates = await collectCandidates()
        var environments: [PythonEnvironment] = []
        var seen = Set<String>()

        for candidate in candidates {
            guard let env = await probe(candidate) else { continue }
            if seen.insert(env.id).inserted {
                environments.append(env)
            }
        }
        return environments
    }

    /// Runs `python -c "import mlx_lm_lora"` against the given executable.
    /// Used by the picker to show a "package missing" badge.
    static func runActivationCheck(executable: String) async -> Bool {
        let result = await runProcess(
            executable: executable,
            arguments: ["-c", "import mlx_lm_lora"],
            timeout: 5
        )
        return result.status == 0
    }

    // MARK: - Candidate collection

    private struct Candidate {
        let path: String
        let kind: PythonEnvironment.Kind
        let nameHint: String?
    }

    private static func collectCandidates() async -> [Candidate] {
        var candidates: [Candidate] = []
        let home = NSHomeDirectory()

        // 1. `which -a python3 python` — anything on PATH.
        let which = await runShell(["-lc", "command -v python3 python 2>/dev/null"])
        for line in which.stdout.split(whereSeparator: \.isNewline) {
            let path = String(line.trimmingCharacters(in: .whitespacesAndNewlines))
            guard !path.isEmpty else { continue }
            candidates.append(Candidate(path: path, kind: inferKind(path: path), nameHint: nil))
        }

        // 2. Common hardcoded locations so we find things even if PATH is sparse.
        for path in [
            "/usr/bin/python3",
            "/usr/local/bin/python3",
            "/opt/homebrew/bin/python3",
            "/opt/homebrew/bin/python",
            "/usr/bin/python",
            "/usr/local/bin/python"
        ] where FileManager.default.isExecutableFile(atPath: path) {
            candidates.append(Candidate(path: path, kind: inferKind(path: path), nameHint: nil))
        }

        // 3. Homebrew Python libexec — what `brew install python3` actually wires
        //    into a real install (`brew --prefix python3/libexec/bin/python3`).
        if let brewPrefix = (await runShell(["-lc", "command -v brew >/dev/null 2>&1 && brew --prefix python3 2>/dev/null"]))
            .stdout.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty {
            let libexecPython = (brewPrefix as NSString).appendingPathComponent("libexec/bin/python3")
            if FileManager.default.isExecutableFile(atPath: libexecPython) {
                candidates.append(Candidate(path: libexecPython, kind: .homebrew, nameHint: "libexec"))
            }
        }

        // 4. pyenv — `$(pyenv root)/versions/*/bin/python3` if pyenv is installed.
        if let pyenvRoot = (await runShell(["-lc", "command -v pyenv >/dev/null 2>&1 && pyenv root 2>/dev/null"]))
            .stdout.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty {
            let versionsDir = (pyenvRoot as NSString).appendingPathComponent("versions")
            if let entries = try? FileManager.default.contentsOfDirectory(atPath: versionsDir) {
                for entry in entries.sorted() {
                    let py = (versionsDir as NSString).appendingPathComponent("\(entry)/bin/python3")
                    if FileManager.default.isExecutableFile(atPath: py) {
                        candidates.append(Candidate(path: py, kind: .pyenv, nameHint: entry))
                    }
                }
            }
        }

        // 5. uv-managed Python installs. A DMG-launched app often has a sparse
        //    PATH, so scan uv's default store directly instead of relying only
        //    on `uv python dir`.
        let uvPythonRoots = [
            "\(home)/.local/share/uv/python",
            "\(home)/Library/Application Support/uv/python"
        ]
        for root in uvPythonRoots {
            collectUVPythons(root: root, candidates: &candidates)
        }

        if let uvPythonDir = (await runShell(["-lc", "for uv in \"$HOME/.local/bin/uv\" /opt/homebrew/bin/uv /usr/local/bin/uv; do [ -x \"$uv\" ] && \"$uv\" python dir 2>/dev/null && exit 0; done; command -v uv >/dev/null 2>&1 && uv python dir 2>/dev/null"]))
            .stdout.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty {
            collectUVPythons(root: uvPythonDir, candidates: &candidates)
        }

        // 6. Conda envs via `conda env list --json`. We try every well-known
        //    `conda` binary so we work for miniconda / anaconda / miniforge /
        //    mambaforge / homebrew conda alike.
        let condaBins = [
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
        for condaBin in condaBins where FileManager.default.isExecutableFile(atPath: condaBin) {
            await collectCondaEnvs(condaBin: condaBin, candidates: &candidates)
        }

        // 7. Project venvs — bounded walk of $HOME (and Desktop / Documents)
        //    to depth 2, skipping directories known to be slow or irrelevant.
        let roots = [home, "\(home)/Desktop", "\(home)/Documents"]
        for root in roots {
            walkForVenvs(in: root, depth: 0, maxDepth: 2, results: &candidates)
        }

        return candidates
    }

    private static func collectUVPythons(root: String, candidates: inout [Candidate]) {
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: root) else { return }
        for entry in entries.sorted() {
            let prefix = (root as NSString).appendingPathComponent(entry)
            for pyName in ["bin/python3", "bin/python"] {
                let py = (prefix as NSString).appendingPathComponent(pyName)
                if FileManager.default.isExecutableFile(atPath: py) {
                    candidates.append(Candidate(path: py, kind: .uv, nameHint: entry))
                    break
                }
            }
        }
    }

    private static func collectCondaEnvs(condaBin: String, candidates: inout [Candidate]) async {
        let result = await runProcess(executable: condaBin, arguments: ["env", "list", "--json"], timeout: 5)
        guard result.status == 0,
              let data = result.stdout.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let envs = json["envs"] as? [String] else { return }

        let condaRoot = (condaBin as NSString).deletingLastPathComponent
        let condaRootURL = URL(fileURLWithPath: condaRoot).deletingLastPathComponent()

        for envPath in envs {
            let py = (envPath as NSString).appendingPathComponent("bin/python")
            guard FileManager.default.isExecutableFile(atPath: py) else { continue }
            let envName = (envPath as NSString).lastPathComponent
            let isBase = (envPath as NSString).standardizingPath == condaRootURL.path
            let hint = isBase ? "base" : envName
            candidates.append(Candidate(path: py, kind: .conda, nameHint: hint))
        }
    }

    private static func walkForVenvs(in dir: String, depth: Int, maxDepth: Int, results: inout [Candidate]) {
        guard depth <= maxDepth else { return }
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: dir) else { return }

        for entry in entries {
            // Skip macOS-managed and other heavy directories.
            if ["Library", "Applications", "Pictures", "Music", "Movies", "Public",
                "node_modules", ".git", ".Trash", "Caches", "Downloads"].contains(entry) {
                continue
            }
            // Skip dot-directories except the venv names we care about.
            if entry.hasPrefix(".") && entry != ".venv" && entry != ".env" { continue }

            let full = (dir as NSString).appendingPathComponent(entry)
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: full, isDirectory: &isDir),
                  isDir.boolValue else { continue }

            // Is this directory itself a venv?
            if ["venv", ".venv", "env", ".env"].contains(entry) {
                for pyName in ["bin/python", "bin/python3"] {
                    let py = (full as NSString).appendingPathComponent(pyName)
                    if FileManager.default.isExecutableFile(atPath: py) {
                        let parent = (dir as NSString).lastPathComponent
                        results.append(Candidate(path: py, kind: .venv, nameHint: parent))
                    }
                }
            }

            walkForVenvs(in: full, depth: depth + 1, maxDepth: maxDepth, results: &results)
        }
    }

    private static func inferKind(path: String) -> PythonEnvironment.Kind {
        let expanded = NSString(string: path).expandingTildeInPath
        if expanded.hasPrefix("/usr/bin/") || expanded.hasPrefix("/usr/lib/")
            || expanded == "/usr/bin/python" || expanded == "/usr/bin/python3" {
            return .system
        }
        if expanded.hasPrefix("/opt/homebrew/") || expanded.hasPrefix("/usr/local/Cellar/") {
            return .homebrew
        }
        if expanded.contains("miniconda3/envs/") || expanded.contains("anaconda3/envs/")
            || expanded.contains("miniforge3/envs/") || expanded.contains("mambaforge/envs/") {
            return .conda
        }
        if expanded.contains("/.pyenv/versions/") {
            return .pyenv
        }
        if expanded.contains("/.local/share/uv/python/") || expanded.contains("/Application Support/uv/python/") {
            return .uv
        }
        return .custom
    }

    // MARK: - Probe

    private static func probe(_ candidate: Candidate) async -> PythonEnvironment? {
        let result = await runProcess(
            executable: candidate.path,
            arguments: ["-c", "import platform; print(platform.python_version())"],
            timeout: 5
        )
        guard result.status == 0 else { return nil }
        let version = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !version.isEmpty else { return nil }

        let id = PythonEnvironment.canonicalID(for: candidate.path)
        let label: String
        if let hint = candidate.nameHint {
            label = "\(candidate.kind.displayName) · \(hint)"
        } else {
            // Derive from path components.
            let parent = ((candidate.path as NSString).deletingLastPathComponent as NSString).lastPathComponent
            let grandparent = ((candidate.path as NSString).deletingLastPathComponent as NSString).deletingLastPathComponent
            let gpName = (grandparent as NSString).lastPathComponent
            label = parent == "bin"
                ? "\(candidate.kind.displayName) · \(gpName)"
                : "\(candidate.kind.displayName) · \(parent)"
        }
        return PythonEnvironment(id: id, path: candidate.path, label: label, version: version, kind: candidate.kind)
    }

    // MARK: - Process helpers

    private struct ShellResult {
        let status: Int32
        let stdout: String
        let stderr: String
    }

    private static func runShell(_ args: [String]) async -> ShellResult {
        await runProcess(executable: "/bin/zsh", arguments: args, timeout: 5)
    }

    /// Spawns a subprocess, reads its stdout / stderr, and returns when the
    /// process exits. Hard-kills the process after `timeout` seconds so a
    /// hung Python can't block the UI scan.
    private static func runProcess(executable: String, arguments: [String], timeout: TimeInterval) async -> ShellResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe

        let timeoutItem = DispatchWorkItem {
            if process.isRunning { process.terminate() }
        }
        DispatchQueue.global().asyncAfter(deadline: .now() + timeout, execute: timeoutItem)

        do {
            try process.run()
        } catch {
            timeoutItem.cancel()
            return ShellResult(status: -1, stdout: "", stderr: "")
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
                timeoutItem.cancel()
                group.notify(queue: .global()) {
                    let stdout = String(data: buffers.stdout, encoding: .utf8) ?? ""
                    let stderr = String(data: buffers.stderr, encoding: .utf8) ?? ""
                    continuation.resume(returning: ShellResult(status: proc.terminationStatus, stdout: stdout, stderr: stderr))
                }
            }
        }
    }
}

/// Mutable buffers shared between the two read pumps and the termination
/// handler. All access is gated by a `DispatchGroup`, so we mark the class
/// `@unchecked Sendable` to tell the compiler we own the synchronization.
private final class PipeBuffers: @unchecked Sendable {
    var stdout = Data()
    var stderr = Data()
}

private extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}
