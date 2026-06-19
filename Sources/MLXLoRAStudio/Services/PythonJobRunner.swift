import Foundation
import Observation

enum RunnerError: LocalizedError {
    case alreadyRunning
    case invalidWorkingDirectory
    case invalidOutputRoot
    case invalidPackagePath
    case invalidSyntheticResumeOutput
    case cannotWriteConfig

    var errorDescription: String? {
        switch self {
        case .alreadyRunning: "A job is already running."
        case .invalidWorkingDirectory: "The working directory does not exist."
        case .invalidOutputRoot: "The output folder could not be created."
        case .invalidPackagePath: "The mlx-lm-lora package path does not exist."
        case .invalidSyntheticResumeOutput: "The synthetic resume folder does not exist."
        case .cannotWriteConfig: "The training config could not be written."
        }
    }
}

@MainActor
@Observable
final class PythonJobRunner {
    var isRunning = false
    var isPaused = false
    var logLines: [String] = []
    var metrics: [TrainingMetric] = []
    var currentCommand = ""
    var lastSpecPath = ""
    var lastRunFolder = ""
    var startedAt: Date?
    var progressCurrent: Int?
    var progressTotal: Int?
    var progressLabel = ""
    private var progressBase = 0

    private var process: Process?
    private var outputPipe: Pipe?
    private let parser = MetricsParser()

    // Debounced writer for `metrics.json`. The trainer emits a steady
    // stream of report lines, so writing on every consume would thrash
    // the disk. The flush is rescheduled on every new sample; the file
    // is only ever written when the run folder is a training folder
    // (i.e. `lastRunFolder` is set) and at least one metric has been
    // recorded.
    @ObservationIgnored private var metricsFlushTask: Task<Void, Never>?
    private static let metricsFlushDelay: Duration = .milliseconds(750)

    func startTraining(
        config: TrainingConfig,
        pythonExecutable: String,
        packagePath: String,
        workingDirectory: String,
        outputRoot: String,
        resourceGuardMemoryPercent: Int,
        huggingFaceToken: String? = nil,
        onCompletion: (@MainActor (Int32) -> Void)? = nil
    ) async throws -> String {
        guard !isRunning else { throw RunnerError.alreadyRunning }
        prepareRun()
        let workURL = URL(fileURLWithPath: workingDirectory)
        guard FileManager.default.fileExists(atPath: workURL.path) else {
            throw RunnerError.invalidWorkingDirectory
        }

        let startDate = Date()
        let runURL = try makeRunFolder(
            outputRoot: outputRoot,
            folderName: config.resolvedRunFolderName(date: startDate)
        )
        let adapterURL = runURL.appending(path: "adapters", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: adapterURL, withIntermediateDirectories: true)
        let specURL = runURL.appending(path: "run_spec.json")
        do {
            try config.runSpecData(adapterPath: adapterURL.path).write(to: specURL, options: .atomic)
        } catch {
            throw RunnerError.cannotWriteConfig
        }
        lastSpecPath = specURL.path
        lastRunFolder = runURL.path
        configureProgress(
            total: config.epochs > 0 ? nil : max(config.iters, 0),
            label: "Training steps"
        )
        appendSystemLine("Run folder: \(runURL.path)")

        let args = [
            "Backend/training_runner.py",
            "--spec", specURL.path
        ]
        return try await launch(
            pythonExecutable: pythonExecutable,
            arguments: args,
            packagePath: packagePath,
            workingDirectory: workingDirectory,
            displayCommand: "Custom \(config.trainMode.title) pipeline",
            huggingFaceToken: huggingFaceToken,
            extraEnvironment: [
                "MLX_LORA_STUDIO_MEMORY_LIMIT_FRACTION": Self.resourceGuardMemoryFractionString(
                    percent: resourceGuardMemoryPercent
                )
            ],
            onCompletion: onCompletion
        )
    }

    func startSynthetic(
        config: SyntheticConfig,
        pythonExecutable: String,
        packagePath: String,
        workingDirectory: String,
        outputRoot: String,
        huggingFaceToken: String? = nil,
        // Keychain-saved API key for the selected provider backend.
        // The store reads this on demand from the macOS Keychain (per
        // provider) and passes it in here so the runner doesn't have
        // to know about Keychain slots. A non-empty value typed into
        // `config.apiKey` by the user (a one-off override) takes
        // precedence over this saved value.
        savedSyntheticAPIKey: String? = nil,
        onCompletion: (@MainActor (Int32) -> Void)? = nil
    ) async throws -> String {
        guard !isRunning else { throw RunnerError.alreadyRunning }
        prepareRun()
        let resumeOutput = config.resumeOutputDir.trimmingCharacters(in: .whitespacesAndNewlines)
        let isResumingOutput = !resumeOutput.isEmpty
        let startDate = Date()
        let runURL: URL
        let dataOutputURL: URL
        if isResumingOutput {
            dataOutputURL = URL(fileURLWithPath: NSString(string: resumeOutput).expandingTildeInPath, isDirectory: true)
            guard FileManager.default.fileExists(atPath: dataOutputURL.path) else {
                throw RunnerError.invalidSyntheticResumeOutput
            }
            runURL = dataOutputURL.deletingLastPathComponent()
        } else {
            runURL = try makeRunFolder(
                outputRoot: outputRoot,
                folderName: config.resolvedRunFolderName(date: startDate)
            )
            dataOutputURL = runURL.appending(path: "generated-data", directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: dataOutputURL, withIntermediateDirectories: true)
        }
        let specURL = runURL.appending(path: "synthetic_spec.json")
        do {
            try config.runSpecData(outputDir: dataOutputURL.path).write(to: specURL, options: .atomic)
        } catch {
            throw RunnerError.cannotWriteConfig
        }
        lastSpecPath = specURL.path
        lastRunFolder = runURL.path
        let existingSyntheticRecords = isResumingOutput
            ? Self.countJSONLRecords(at: dataOutputURL.appending(path: "output_full.jsonl"))
            : 0
        configureProgress(
            total: max(config.numSamples, 0),
            label: "Synthetic generation",
            initial: existingSyntheticRecords
        )
        if existingSyntheticRecords > 0 {
            appendSystemLine("Continuing synthetic progress from \(existingSyntheticRecords) existing records.")
        }
        appendSystemLine("Run folder: \(runURL.path)")

        let subcommand = config.kind == .sft ? "synthetic_sft" : "synthetic_dpo"
        var args = ["-m", "mlx_lm_lora", subcommand]
        args += ["--dataset-path", config.datasetPath]
        if config.kind == .sft {
            args += ["--model", config.model]
            args += ["--backend", config.backend.rawValue]
            if !config.baseURL.isEmpty { args += ["--base-url", config.baseURL] }
            if config.multiturn { args.append("--multiturn") }
            args += ["--max-turns", "\(config.maxTurns)"]
            args += ["--max-concurrent", "\(config.maxConcurrent)"]
            args += ["--multiturn-percentile", "\(config.multiturnPercentile)"]
            if !config.humanRoleModel.isEmpty { args += ["--human-role-model", config.humanRoleModel] }
            if config.includeSystemPrompt { args.append("--include-system-prompt") }
            args.append(config.useGroundTruth ? "--use-ground-truth" : "--no-use-ground-truth")
        } else {
            args += ["--base-model", config.baseModel, "--teacher-model", config.teacherModel]
            args += ["--generation-target", config.dpoGenerationTarget.rawValue]
            args += ["--backend", config.backend.rawValue]
            if !config.baseURL.isEmpty { args += ["--base-url", config.baseURL] }
            args += ["--max-concurrent", "\(config.maxConcurrent)"]
        }
        args += ["--output-dir", dataOutputURL.path]
        if isResumingOutput { args.append("--resume-output") }
        if !config.systemPrompt.isEmpty { args += ["--system-prompt", config.systemPrompt] }
        args += ["--num-samples", "\(config.numSamples)"]
        if !config.validSplit.isEmpty { args += ["--valid-split", config.validSplit] }
        if !config.testSplit.isEmpty { args += ["--test-split", config.testSplit] }
        args += ["--batch-size", "\(config.batchSize)"]
        // The Python CLI declares this flag with
        // `argparse.BooleanOptionalAction`, so argparse auto-generates
        // the negation as `--no-use-generation-settings` (NOT
        // `--no-generation-settings`). Sending the wrong form makes
        // the trainer die with `unrecognized arguments` before the
        // run even starts. Both SFT and DPO honor the same flag.
        args.append(config.useGenerationSettings ? "--use-generation-settings" : "--no-use-generation-settings")
        if config.useGenerationSettings {
            args += ["--max-tokens", "\(config.maxTokens)"]
            args += ["--temperature", "\(config.temperature)"]
            args += ["--top-p", "\(config.topP)"]
            args += ["--min-p", "\(config.minP)"]
            args += ["--top-k", "\(config.topK)"]
            args += ["--min-tokens-to-keep", "\(config.minTokensToKeep)"]
            args += ["--xtc-probability", "\(config.xtcProbability)"]
            args += ["--xtc-threshold", "\(config.xtcThreshold)"]
        }
        args += ["--seed", "\(config.seed)"]

        // The form-typed `config.apiKey` always wins (it's a one-off
        // override the user just pasted in). If the user cleared the
        // field, fall back to the saved key from the Keychain slot
        // for the currently-selected backend. Either way we inject
        // the result as a synthetic-specific env var so we never
        // touch the parent environment's `OPENAI_API_KEY` /
        // `OPENROUTER_API_KEY` / etc. — the user may have those set
        // globally for other tools and we don't want to interfere.
        let resolvedKey = config.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        let effectiveKey = resolvedKey.isEmpty
            ? (savedSyntheticAPIKey?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "")
            : resolvedKey
        var extraEnv: [String: String] = [:]
        if !effectiveKey.isEmpty {
            extraEnv["SYNTHETIC_OPENAI_API_KEY"] = effectiveKey
        }

        return try await launch(
            pythonExecutable: pythonExecutable,
            arguments: args,
            packagePath: packagePath,
            workingDirectory: workingDirectory,
            huggingFaceToken: huggingFaceToken,
            extraEnvironment: extraEnv,
            onCompletion: onCompletion
        )
    }

    func startHFUpload(
        config: HFUploadConfig,
        target: HFUploadTarget = .all,
        pythonExecutable: String,
        packagePath: String,
        workingDirectory: String,
        outputRoot: String,
        huggingFaceToken: String? = nil,
        onCompletion: (@MainActor (Int32) -> Void)? = nil
    ) async throws -> String {
        guard !isRunning else { throw RunnerError.alreadyRunning }
        prepareRun()

        let startDate = Date()
        let runURL = try makeRunFolder(
            outputRoot: outputRoot,
            folderName: RunFolderNamer.makeName(pieces: ["hf-upload"], date: startDate)
        )
        let specURL = runURL.appending(path: "hf_upload_spec.json")
        do {
            try config.runSpecData(target: target).write(to: specURL, options: .atomic)
        } catch {
            throw RunnerError.cannotWriteConfig
        }
        lastSpecPath = specURL.path
        lastRunFolder = runURL.path
        configureProgress(total: nil, label: "Upload")
        appendSystemLine("Upload spec: \(specURL.path)")

        let args = [
            "Backend/hf_upload.py",
            "--spec", specURL.path
        ]
        return try await launch(
            pythonExecutable: pythonExecutable,
            arguments: args,
            packagePath: packagePath,
            workingDirectory: workingDirectory,
            displayCommand: "Upload \(target.title) to Hugging Face",
            huggingFaceToken: huggingFaceToken,
            onCompletion: onCompletion
        )
    }

    func startPackageUpdate(
        pythonExecutable: String,
        packagePath: String,
        workingDirectory: String,
        forceReinstall: Bool = false,
        onCompletion: (@MainActor (Int32) -> Void)? = nil
    ) async throws -> String {
        guard !isRunning else { throw RunnerError.alreadyRunning }
        prepareRun()

        let workURL = URL(fileURLWithPath: workingDirectory)
        guard FileManager.default.fileExists(atPath: workURL.path) else {
            throw RunnerError.invalidWorkingDirectory
        }

        let packageURL = URL(fileURLWithPath: packagePath)
        guard FileManager.default.fileExists(atPath: packageURL.path) else {
            throw RunnerError.invalidPackagePath
        }

        let installVerb = forceReinstall ? "Force-reinstalling" : "Updating"
        configureProgress(total: nil, label: forceReinstall ? "Package reinstall" : "Package update")
        let reinstallFlag = forceReinstall ? "--force-reinstall" : ""
        let reinstallArgs = reinstallFlag.isEmpty ? "" : "\(reinstallFlag) "
        let script = """
        set -e
        PYTHON_BIN=\(ShellQuote.escape(pythonExecutable))
        PACKAGE_DIR=\(ShellQuote.escape(packageURL.path))
        UV_BIN=""
        for candidate in "$HOME/.local/bin/uv" "/opt/homebrew/bin/uv" "/usr/local/bin/uv"; do
          if [ -x "$candidate" ]; then
            UV_BIN="$candidate"
            break
          fi
        done
        if [ -z "$UV_BIN" ] && command -v uv >/dev/null 2>&1; then
          UV_BIN="$(command -v uv)"
        fi

        PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        case "$PY_VERSION" in
          3.14|3.15|3.16|3.17|3.18|3.19)
            echo "Warning: Python $PY_VERSION is very new; mlx/mlx-lm wheels may not be published for it yet."
            echo "If installation fails, create/select a Python 3.12 or 3.13 environment and retry."
            ;;
        esac

        pip_install() {
          if [ -n "$UV_BIN" ]; then
            "$UV_BIN" pip install --python "$PYTHON_BIN" "$@"
          else
            "$PYTHON_BIN" -m pip install "$@"
          fi
        }

        echo "Updating mlx-lm-lora checkout..."
        if [ -d "\(ShellQuote.escape(packageURL.appending(path: ".git").path))" ]; then
          git -C "$PACKAGE_DIR" pull --ff-only
        else
          echo "No git checkout found at $PACKAGE_DIR; skipping git pull."
        fi
        if [ -n "$UV_BIN" ]; then
          echo "Using uv at $UV_BIN with interpreter $PYTHON_BIN."
        else
          echo "uv not found; falling back to $PYTHON_BIN -m pip."
          echo "Updating pip, setuptools, and wheel..."
          pip_install -U pip setuptools wheel
        fi
        echo "\(installVerb) mlx, mlx-lm, and matplotlib..."
        pip_install -U \(reinstallArgs)mlx mlx-lm matplotlib
        echo "\(installVerb) local mlx-lm-lora package..."
        pip_install -U \(reinstallArgs)-e "$PACKAGE_DIR"
        echo "Package \(forceReinstall ? "reinstall" : "update") complete."
        """

        return try await launchShell(
            script: script,
            packagePath: packagePath,
            workingDirectory: workingDirectory,
            displayCommand: forceReinstall
                ? "Reinstall mlx, mlx-lm, matplotlib, and mlx-lm-lora"
                : "Update mlx, mlx-lm, matplotlib, and mlx-lm-lora",
            onCompletion: onCompletion
        )
    }

    private func makeRunFolder(outputRoot: String, folderName: String) throws -> URL {
        let expandedRoot = NSString(string: outputRoot).expandingTildeInPath
        let rootURL = URL(fileURLWithPath: expandedRoot, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
            let runURL = uniqueRunURL(rootURL: rootURL, folderName: folderName)
            try FileManager.default.createDirectory(at: runURL, withIntermediateDirectories: true)
            return runURL
        } catch {
            throw RunnerError.invalidOutputRoot
        }
    }

    private func uniqueRunURL(rootURL: URL, folderName: String) -> URL {
        let cleanName = RunFolderNamer.sanitize(folderName)
        var candidate = rootURL.appending(path: cleanName, directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: candidate.path) else { return candidate }
        for index in 2...999 {
            candidate = rootURL.appending(path: "\(cleanName)-\(index)", directoryHint: .isDirectory)
            if !FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
        }
        return rootURL.appending(path: "\(cleanName)-\(UUID().uuidString)", directoryHint: .isDirectory)
    }

    private func prepareRun() {
        logLines.removeAll()
        metrics.removeAll()
        lastSpecPath = ""
        lastRunFolder = ""
        metricsFlushTask?.cancel()
        metricsFlushTask = nil
        parser.reset()
        startedAt = nil
        progressCurrent = nil
        progressTotal = nil
        progressLabel = ""
        progressBase = 0
    }

    private func configureProgress(
        total: Int?,
        label: String,
        initial: Int = 0
    ) {
        progressTotal = total.flatMap { $0 > 0 ? $0 : nil }
        progressBase = max(initial, 0)
        if let progressTotal {
            progressCurrent = min(progressBase, progressTotal)
        } else {
            progressCurrent = nil
        }
        progressLabel = label
    }

    func stop() {
        process?.terminate()
        process = nil
        isRunning = false
        isPaused = false
        finishProgressIfPossible()
        flushMetricsNow()
        appendSystemLine("Job stopped.")
    }

    func pause() {
        guard isRunning, !isPaused, let process else { return }
        Darwin.kill(process.processIdentifier, SIGSTOP)
        isPaused = true
        appendSystemLine("Job paused.")
    }

    func resume() {
        guard isRunning, isPaused, let process else { return }
        Darwin.kill(process.processIdentifier, SIGCONT)
        isPaused = false
        appendSystemLine("Job resumed.")
    }

    func appendSystemLine(_ line: String) {
        logLines.append("[Studio] \(line)")
    }

    func clearTerminal() {
        logLines.removeAll()
        appendSystemLine("Terminal cleared.")
    }

    private func launch(
        pythonExecutable: String,
        arguments: [String],
        packagePath: String,
        workingDirectory: String,
        displayCommand: String? = nil,
        huggingFaceToken: String? = nil,
        extraEnvironment: [String: String] = [:],
        onCompletion: (@MainActor (Int32) -> Void)? = nil
    ) async throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = arguments
        process.currentDirectoryURL = URL(fileURLWithPath: workingDirectory)

        var environment = ProcessInfo.processInfo.environment
        let existingPath = environment["PYTHONPATH"].map { ":\($0)" } ?? ""
        environment["PYTHONPATH"] = packagePath + existingPath
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONWARNINGS"] = PythonWarningFilter.merging(
            "ignore:resource_tracker:UserWarning",
            into: environment["PYTHONWARNINGS"]
        )
        HuggingFaceEnvironment.apply(huggingFaceToken, to: &environment)
        for (key, value) in extraEnvironment {
            environment[key] = value
        }
        process.environment = environment

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        self.process = process
        outputPipe = pipe

        let command = displayCommand ?? ([pythonExecutable] + arguments).joined(separator: " ")
        currentCommand = command
        appendSystemLine("Launching \(command)")

        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor [weak self] in
                self?.consume(text)
            }
        }

        process.terminationHandler = { [weak self] process in
            Task { @MainActor [weak self] in
                self?.isRunning = false
                self?.isPaused = false
                self?.finishProgressIfPossible()
                self?.appendSystemLine("Job exited with status \(process.terminationStatus).")
                self?.outputPipe?.fileHandleForReading.readabilityHandler = nil
                self?.process = nil
                self?.outputPipe = nil
                // The on-disk metrics file is the source of truth for
                // the Runs page after the process exits, so write it
                // synchronously before we let the run hand back to the
                // store.
                self?.flushMetricsNow()
                onCompletion?(process.terminationStatus)
            }
        }

        try process.run()
        isRunning = true
        isPaused = false
        startedAt = Date()
        return command
    }

    private func launchShell(
        script: String,
        packagePath: String,
        workingDirectory: String,
        displayCommand: String,
        huggingFaceToken: String? = nil,
        onCompletion: (@MainActor (Int32) -> Void)? = nil
    ) async throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-lc", script]
        process.currentDirectoryURL = URL(fileURLWithPath: workingDirectory)

        var environment = ProcessInfo.processInfo.environment
        let existingPath = environment["PYTHONPATH"].map { ":\($0)" } ?? ""
        environment["PYTHONPATH"] = packagePath + existingPath
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONWARNINGS"] = PythonWarningFilter.merging(
            "ignore:resource_tracker:UserWarning",
            into: environment["PYTHONWARNINGS"]
        )
        HuggingFaceEnvironment.apply(huggingFaceToken, to: &environment)
        process.environment = environment

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        self.process = process
        outputPipe = pipe

        currentCommand = displayCommand
        appendSystemLine("Launching \(displayCommand)")

        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor [weak self] in
                self?.consume(text)
            }
        }

        process.terminationHandler = { [weak self] process in
            Task { @MainActor [weak self] in
                self?.isRunning = false
                self?.isPaused = false
                self?.finishProgressIfPossible()
                self?.appendSystemLine("Package update exited with status \(process.terminationStatus).")
                self?.outputPipe?.fileHandleForReading.readabilityHandler = nil
                self?.process = nil
                self?.outputPipe = nil
                self?.flushMetricsNow()
                onCompletion?(process.terminationStatus)
            }
        }

        try process.run()
        isRunning = true
        isPaused = false
        startedAt = Date()
        return displayCommand
    }

    private static func resourceGuardMemoryFractionString(percent: Int) -> String {
        let clamped = min(max(percent, 10), 98)
        return String(format: "%.2f", Double(clamped) / 100.0)
    }

    private func consume(_ text: String) {
        let cleaned = text.replacingOccurrences(of: "\r", with: "\n")
        var newMetricCount = 0
        for rawLine in cleaned.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(rawLine)
            let plainLine = ANSIText.clean(line)
            guard !plainLine.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
            guard !TerminalNoiseFilter.shouldSuppress(plainLine) else { continue }
            // The parser is stateful so it can buffer a GRPO multi-line
            // block across calls; a `nil` return just means "this line
            // is part of a block I'm still assembling, or it's noise".
            for metric in parser.consume(plainLine) {
                updateProgress(step: metric.step)
                // The trainer emits two lines per report: a tqdm progress
                // bar (`... loss: 2.444, it/s: 0.81]`) and a `tqdm.write`
                // `Iter N: loss …, lr …, it/s …, tok/s …, peak_mem …`
                // line. Each carries a different subset of keys. We merge
                // metrics that share a step so the user sees a complete
                // picture instead of whichever line arrived last.
                if let lastIndex = metrics.lastIndex(where: { $0.step == metric.step }),
                   !metric.values.isEmpty {
                    let existing = metrics[lastIndex]
                    var merged = existing.values
                    for (k, v) in metric.values { merged[k] = v }
                    metrics[lastIndex] = TrainingMetric(
                        step: existing.step,
                        values: merged,
                        rawLine: existing.rawLine + "\n" + metric.rawLine
                    )
                    newMetricCount += 1
                } else {
                    metrics.append(metric)
                    newMetricCount += 1
                }
            }
            updateProgress(fromLogLine: plainLine)
            if !plainLine.hasPrefix("@@studio_metric ") {
                logLines.append(plainLine)
            }
        }
        if logLines.count > 1500 {
            logLines.removeFirst(logLines.count - 1500)
        }
        if newMetricCount > 0 {
            scheduleMetricsFlush()
        }
    }

    private func updateProgress(current: Int) {
        guard let total = progressTotal else { return }
        progressCurrent = min(max(current, progressCurrent ?? 0), total)
    }

    private func updateProgress(step: Int) {
        updateProgress(current: step)
    }

    private func updateProgress(fromLogLine line: String) {
        updateStepProgressMetadata(fromLogLine: line)
        if let parsed = Self.tqdmProgress(in: line) {
            if parsed.total > 0 {
                let parsedTotal = progressBase + parsed.total
                progressTotal = max(progressTotal ?? 0, parsedTotal)
            }
            updateProgress(current: progressBase + parsed.current)
            return
        }

        let savedPattern = #"Saved\s+(\d+)\s+(?:SFT\s+)?examples"#
        if let count = Self.firstRegexInt(in: line, pattern: savedPattern) {
            updateProgress(current: count)
        }
    }

    private func updateStepProgressMetadata(fromLogLine line: String) {
        let pattern = #"Calculated\s+(\d+)\s+iterations\s+from\s+(\d+)\s+epochs"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return }
        let range = NSRange(line.startIndex..<line.endIndex, in: line)
        guard let match = regex.firstMatch(in: line, range: range),
              match.numberOfRanges >= 3,
              let iterationsRange = Range(match.range(at: 1), in: line),
              let epochsRange = Range(match.range(at: 2), in: line),
              let iterations = Int(line[iterationsRange]),
              let epochs = Int(line[epochsRange]),
              iterations > 0,
              epochs > 0 else {
            return
        }
        progressTotal = iterations
        progressCurrent = min(max(progressCurrent ?? 0, 0), iterations)
    }

    private func finishProgressIfPossible() {
        if let total = progressTotal, total > 0 {
            progressCurrent = min(max(progressCurrent ?? 0, 0), total)
        }
    }

    private static func tqdmProgress(in line: String) -> (current: Int, total: Int)? {
        let pattern = #"(\d+)\s*/\s*(\d+)"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(line.startIndex..<line.endIndex, in: line)
        guard let match = regex.firstMatch(in: line, range: range),
              match.numberOfRanges >= 3,
              let currentRange = Range(match.range(at: 1), in: line),
              let totalRange = Range(match.range(at: 2), in: line),
              let current = Int(line[currentRange]),
              let total = Int(line[totalRange]) else {
            return nil
        }
        return (current, total)
    }

    private static func firstRegexInt(in line: String, pattern: String) -> Int? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return nil }
        let range = NSRange(line.startIndex..<line.endIndex, in: line)
        guard let match = regex.firstMatch(in: line, range: range),
              match.numberOfRanges >= 2,
              let valueRange = Range(match.range(at: 1), in: line) else {
            return nil
        }
        return Int(line[valueRange])
    }

    private static func countJSONLRecords(at url: URL) -> Int {
        guard let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .utf8) else {
            return 0
        }
        return text.split(separator: "\n", omittingEmptySubsequences: true).count
    }

    // MARK: - Metrics persistence

    /// Reschedule a debounced write of `metrics` to
    /// `<lastRunFolder>/metrics.json`. Cancels any in-flight write
    /// first, so a torrent of trainer output collapses to a single
    /// disk hit per quiet period.
    private func scheduleMetricsFlush() {
        guard !lastRunFolder.isEmpty else { return }
        metricsFlushTask?.cancel()
        metricsFlushTask = Task { [weak self] in
            try? await Task.sleep(for: Self.metricsFlushDelay)
            if Task.isCancelled { return }
            await MainActor.run {
                self?.flushMetricsNow()
            }
        }
    }

    /// Write `metrics` to disk synchronously. Safe to call repeatedly
    /// (each write is atomic) and from the termination handler
    /// (we're already on the main actor at that point).
    private func flushMetricsNow() {
        metricsFlushTask?.cancel()
        metricsFlushTask = nil
        guard !lastRunFolder.isEmpty, !metrics.isEmpty else { return }
        let folder = URL(fileURLWithPath: lastRunFolder, isDirectory: true)
        let url = folder.appending(path: TrainingMetricIO.filename)
        do {
            try TrainingMetricIO.write(metrics, to: url)
        } catch {
            appendSystemLine("Could not write metrics.json: \(error.localizedDescription)")
        }
    }
}

enum ShellQuote {
    static func escape(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}

/// Injects a Hugging Face personal access token into a child-process
/// environment dict as both `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` —
/// the `huggingface_hub` library checks either name, so setting both
/// makes the runner robust to whichever the trainer code path reads.
/// An empty / nil token is treated as "no token": both env vars are
/// removed so a stale token from the parent environment does not
/// leak into the child.
enum HuggingFaceEnvironment {
    static func apply(_ token: String?, to environment: inout [String: String]) {
        let trimmed = token?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if trimmed.isEmpty {
            environment.removeValue(forKey: "HF_TOKEN")
            environment.removeValue(forKey: "HUGGING_FACE_HUB_TOKEN")
        } else {
            environment["HF_TOKEN"] = trimmed
            environment["HUGGING_FACE_HUB_TOKEN"] = trimmed
        }
    }
}

enum PythonWarningFilter {
    static func merging(_ filter: String, into existing: String?) -> String {
        guard let existing, !existing.isEmpty else { return filter }
        guard !existing.split(separator: ",").contains(where: { $0 == filter }) else {
            return existing
        }
        return existing + "," + filter
    }
}

enum TerminalNoiseFilter {
    static func shouldSuppress(_ line: String) -> Bool {
        let lower = line.lowercased()
        return lower.contains("multiprocessing/resource_tracker.py")
            || lower.contains("multiprocessing\\resource_tracker.py")
            || lower.contains("resource_tracker.py:")
            || lower.contains("resource_tracker: there appear to be")
            || lower.contains("leaked semaphore objects to clean up at shutdown")
            || lower.contains("warnings.warn('resource_tracker:")
    }
}

enum ANSIText {
    static func clean(_ text: String) -> String {
        text.replacingOccurrences(
            of: #"\u{001B}\[[0-?]*[ -/]*[@-~]"#,
            with: "",
            options: .regularExpression
        )
    }
}

/// Streaming parser for trainer stdout. Three report formats are supported:
///
/// 0. **Studio JSON** emitted by `Backend/training_runner.py` callbacks:
///    ```
///    @@studio_metric {"event":"train","iteration":10,"train_loss":1.2}
///    ```
///
/// 1. **Inline** (SFT, DPO, CPO, ORPO, PPO, Online-DPO, XPO, RLHF Reinforce):
///    ```
///    Iter 150: Train loss 2.011, lr 1.000e-05, it/s 0.675, tok/s 1259.263, peak_mem 7.287GB
///    ```
///    One line, several `key value` chunks joined by `,`.
///
/// 2. **GRPO block** (delimited by `========`):
///    ```
///    ========================================
///    Iter 30:
///    ----------------------------------------
///    Loss: 2.011
///    Total Rewards:  μ=0.512, σ=0.123
///    • helpfulness: μ=0.621, σ=0.142, cov=87.50%
///    ...
///    ========================================
///    ```
///    Multiple lines, parsed together. The per-reward `•` lines are
///    expanded into three keys (`reward_<name>_mean`, `..._std`, `..._coverage`).
///
/// `consume(_:)` is stateful: a GRPO block may be split across many
/// reads of the pipe, so we accumulate lines until we see the closing
/// `=========` and then emit the assembled metric. `reset()` clears the
/// accumulator and is called at the start of every new job.
final class MetricsParser {
    private var grpoBlock: [String] = []
    private var inGrpoBlock = false

    func reset() {
        grpoBlock.removeAll(keepingCapacity: true)
        inGrpoBlock = false
    }

    /// Feed a single line (already ANSI-stripped). Returns 0 or 1 metrics.
    func consume(_ line: String) -> [TrainingMetric] {
        let plain = ANSIText.clean(line)
        let trimmed = plain.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return [] }

        if let metric = parseStudioMetric(trimmed) {
            return [metric]
        }

        // GRPO reports are wrapped in 80-char `=` lines. We accumulate
        // everything between the opening and closing `===` line and
        // only parse once we have the full block.
        if trimmed.hasPrefix("=======") {
            if !inGrpoBlock {
                inGrpoBlock = true
                grpoBlock.removeAll(keepingCapacity: true)
            } else {
                inGrpoBlock = false
                let block = grpoBlock
                grpoBlock.removeAll(keepingCapacity: true)
                if let metric = parseGrpoBlock(block) {
                    return [metric]
                }
                return []
            }
            return []
        }
        if inGrpoBlock {
            grpoBlock.append(plain)
            return []
        }

        if let metric = parseInline(plain) {
            return [metric]
        }
        return []
    }

    // MARK: - Studio JSON parser

    private func parseStudioMetric(_ line: String) -> TrainingMetric? {
        let prefix = "@@studio_metric "
        guard line.hasPrefix(prefix) else { return nil }
        let jsonText = String(line.dropFirst(prefix.count))
        guard let data = jsonText.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }

        let step = intValue(object["iteration"]) ?? 0
        var values: [String: Double] = [:]
        for (key, value) in object {
            guard key != "event", key != "iteration",
                  let doubleValue = doubleValue(value) else { continue }
            values[canonicalMetricKey(key)] = doubleValue
        }

        guard !values.isEmpty else { return nil }
        return TrainingMetric(step: step, values: values, rawLine: line)
    }

    private func canonicalMetricKey(_ key: String) -> String {
        switch key {
        case "train_loss": return "loss"
        case "train_chosen_reward": return "chosen_r"
        case "train_rejected_reward": return "rejected_r"
        case "learning_rate": return "learning_rate"
        case "iterations_per_second": return "it_s"
        case "tokens_per_second": return "tok_s"
        case "trained_tokens": return "trained_tok"
        case "peak_memory": return "peak_mem"
        default:
            if key.hasPrefix("train_") {
                return String(key.dropFirst("train_".count))
            }
            return key
        }
    }

    // MARK: - Inline parser

    private func parseInline(_ line: String) -> TrainingMetric? {
        let lower = line.lowercased()
        // Inline trainers always mention `loss`, `peak_mem`/`peak mem`,
        // or `it/s`. GRPO-style lines shouldn't reach here because the
        // block detector above intercepts them first.
        func has(_ needle: String) -> Bool {
            lower.range(of: needle) != nil
        }
        let isTrainerLine =
            has("loss") || has("peak_mem") || has("peak mem") ||
            has("peak memory") || has("memory:") || has("it/s")
        guard isTrainerLine else { return nil }

        let step = firstInt(afterAnyOf: ["iter", "step"], in: lower) ?? 0
        var values: [String: Double] = [:]

        // Loss (training / validation / final test)
        let isValidationLine = lower.contains("| validation |")
        let isTestLine = lower.contains("| test |")
        if let v = firstDouble(afterAnyOf: ["val loss", "validation loss"], in: lower) {
            values["val_loss"] = v
        } else if isValidationLine, let v = firstDouble(afterAnyOf: ["loss"], in: lower) {
            values["val_loss"] = v
        }
        if let v = firstDouble(afterAnyOf: ["test loss"], in: lower) {
            values["test_loss"] = v
        } else if isTestLine, let v = firstDouble(afterAnyOf: ["loss"], in: lower) {
            values["test_loss"] = v
        }
        if let v = firstDouble(afterAnyOf: ["train loss"], in: lower) {
            values["loss"] = v
        } else if !isValidationLine && !isTestLine,
                  let v = firstDouble(afterAnyOf: ["loss"], in: lower) {
            if values["loss"] == nil { values["loss"] = v }
        }

        // Validation metrics
        if let v = firstDouble(afterAnyOf: ["val chosen reward"], in: lower) {
            values["val_chosen_r"] = v
        }
        if let v = firstDouble(afterAnyOf: ["val rejected reward"], in: lower) {
            values["val_rejected_r"] = v
        }
        if let v = firstDouble(afterAnyOf: ["val rewards", "val reward"], in: lower) {
            values["val_rewards"] = v
        }
        if let v = firstDouble(afterAnyOf: ["val kl penalty"], in: lower) {
            values["val_kl_penalty"] = v
        }
        if let v = firstDouble(afterAnyOf: ["val advantages"], in: lower) {
            values["val_advantages"] = v
        }

        // Learning rate
        if let v = firstDouble(afterAnyOf: ["learning rate"], in: lower) {
            values["learning_rate"] = v
        } else if let v = firstDouble(afterAnyOf: ["lr"], in: lower) {
            if values["learning_rate"] == nil { values["learning_rate"] = v }
        }

        // Memory — multiple spellings across trainers
        if let v = firstDouble(afterAnyOf: ["peak_mem", "peak mem", "peak memory", "memory:"], in: lower) {
            values["peak_mem"] = v
        }

        // Throughput
        if let v = firstDouble(afterAnyOf: ["it/s", "iterations/s", "it /s"], in: lower) {
            values["it_s"] = v
        }
        if let v = firstDouble(afterAnyOf: ["tok/s", "tokens/s", "tok /s"], in: lower) {
            values["tok_s"] = v
        }

        // Preference rewards (DPO/CPO/ORPO/PPO/Online-DPO/XPO)
        if let v = firstDouble(afterAnyOf: ["chosen_r", "chosen r"], in: lower) {
            values["chosen_r"] = v
        }
        if let v = firstDouble(afterAnyOf: ["rejected_r", "rejected r"], in: lower) {
            values["rejected_r"] = v
        }

        // RLHF Reinforce — guard `rewards` against `total_rewards_*` etc.
        if let v = firstDouble(afterAnyOf: ["rewards"], in: lower) {
            let prev = charBefore("rewards", in: lower)
            if !isAlnum(prev) {
                if values["rewards"] == nil { values["rewards"] = v }
            }
        }
        if let v = firstDouble(afterAnyOf: ["kl penalty", "kl_penalty"], in: lower) {
            values["kl_penalty"] = v
        }
        if let v = firstDouble(afterAnyOf: ["advantages"], in: lower) {
            if values["advantages"] == nil { values["advantages"] = v }
        }

        // Trained tokens (informational)
        if let v = firstDouble(afterAnyOf: ["trained_tok", "trained tok"], in: lower) {
            values["trained_tok"] = v
        }

        guard !values.isEmpty else { return nil }
        return TrainingMetric(step: step, values: values, rawLine: line)
    }

    // MARK: - GRPO block parser

    private func parseGrpoBlock(_ lines: [String]) -> TrainingMetric? {
        guard !lines.isEmpty else { return nil }
        let joined = lines.joined(separator: "\n")
        let lower = joined.lowercased()
        let step = firstInt(afterAnyOf: ["iter"], in: lower) ?? 0

        var values: [String: Double] = [:]

        // Common `Key: value` lines.
        let keyValueMap: [(key: String, aliases: [String])] = [
            ("loss",                  ["loss:"]),
            ("learning_rate",         ["learning rate:"]),
            ("peak_mem",              ["memory:"]),
            ("kl",                    ["kl divergence:", "kl:"]),
            ("avg_generated_tokens",  ["avg tokens:"]),
            ("min_generated_tokens",  ["min tokens:"]),
            ("max_generated_tokens",  ["max tokens:"]),
            ("hit_max_tokens_ratio",  ["hit limit:"]),
        ]
        for entry in keyValueMap {
            if let v = firstDoubleAfterAliases(entry.aliases, in: joined) {
                values[entry.key] = v
            }
        }

        // `μ=X.XXX, σ=Y.YYY` and `cov=Z.ZZ%` sub-keys. Per-reward lines
        // become `reward_<name>_*` keys; top-level lines become
        // `<key>_mean` / `<key>_std`.
        for line in lines {
            let trimmedLine = line.trimmingCharacters(in: .whitespaces)
            if let rewardName = parsePerRewardName(from: trimmedLine) {
                if let (mean, std, cov) = parseMuSigmaCov(in: trimmedLine) {
                    values["reward_\(rewardName)_mean"] = mean
                    values["reward_\(rewardName)_std"] = std
                    if let c = cov { values["reward_\(rewardName)_coverage"] = c }
                }
                continue
            }
            if let (key, mean, std) = parseTopLevelMuSigma(in: trimmedLine) {
                values["\(key)_mean"] = mean
                if let s = std { values["\(key)_std"] = s }
            }
        }

        // `Speed:` and `Clipping:` lines have an unusual layout.
        for line in lines {
            let lineLower = line.lowercased()
            if lineLower.contains("speed:") {
                if let v = firstDouble(afterAnyOf: ["it/s", "iterations/s"], in: lineLower) {
                    values["it_s"] = v
                }
                if let v = firstDouble(afterAnyOf: ["tok/s", "tokens/s"], in: lineLower) {
                    values["tok_s"] = v
                }
            }
            if lineLower.contains("clipping:") {
                if let v = firstDouble(afterAnyOf: ["low="], in: lineLower) {
                    values["clip_ratio_low"] = v
                }
                if let v = firstDouble(afterAnyOf: ["high="], in: lineLower) {
                    values["clip_ratio_high"] = v
                }
                if let v = firstDouble(afterAnyOf: ["total="], in: lineLower) {
                    values["clip_ratio_total"] = v
                }
            }
        }

        guard !values.isEmpty else { return nil }
        return TrainingMetric(step: step, values: values, rawLine: joined)
    }

    // MARK: - Sub-parsers

    /// Extract the `name` from a per-reward line `• name: μ=…`.
    private func parsePerRewardName(from line: String) -> String? {
        guard line.hasPrefix("•") else { return nil }
        let body = line.dropFirst().trimmingCharacters(in: .whitespaces)
        guard let colonIndex = body.firstIndex(of: ":") else { return nil }
        let name = body[..<colonIndex]
            .trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: #"\s+"#, with: "_", options: .regularExpression)
            .replacingOccurrences(of: #"[^A-Za-z0-9_]"#, with: "", options: .regularExpression)
        return name.isEmpty ? nil : name
    }

    /// Parse `μ=X.XXX, σ=Y.YYY, cov=Z.ZZ%` from a line.
    private func parseMuSigmaCov(in line: String) -> (Double, Double, Double?)? {
        let muPattern = #"μ\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?)"#
        let sigmaPattern = #"σ\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?)"#
        let covPattern = #"cov\s*=\s*([-+]?[0-9]*\.?[0-9]+)\s*%?"#

        guard let mean = firstMatch(in: line, pattern: muPattern) else { return nil }
        guard let std = firstMatch(in: line, pattern: sigmaPattern) else { return nil }
        let cov = firstMatch(in: line, pattern: covPattern).map { $0 / 100.0 }
        return (mean, std, cov)
    }

    /// Parse a top-level line like `Total Rewards:  μ=0.512, σ=0.123`.
    private func parseTopLevelMuSigma(in line: String) -> (String, Double, Double?)? {
        guard !line.hasPrefix("•") else { return nil }
        guard let colonIndex = line.firstIndex(of: ":") else { return nil }
        let label = line[..<colonIndex]
            .trimmingCharacters(in: .whitespaces)
            .lowercased()
        guard line.contains("μ=") else { return nil }
        let key = label
            .replacingOccurrences(of: "μ", with: "")
            .replacingOccurrences(of: #"[^A-Za-z0-9]+"#, with: "_", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        guard !key.isEmpty else { return nil }
        guard let (mean, std, _) = parseMuSigmaCov(in: line) else { return nil }
        return (key, mean, std)
    }

    // MARK: - Number extraction helpers

    private func firstInt(afterAnyOf keys: [String], in text: String) -> Int? {
        firstNumber(afterAnyOf: keys, in: text).map { Int($0) }
    }

    private func firstDouble(afterAnyOf keys: [String], in text: String) -> Double? {
        firstNumber(afterAnyOf: keys, in: text)
    }

    private func firstNumber(afterAnyOf keys: [String], in text: String) -> Double? {
        for key in keys {
            guard let keyRange = text.range(of: key) else { continue }
            let tail = text[keyRange.upperBound...]
            let pattern = #"[-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?"#
            if let range = tail.range(of: pattern, options: .regularExpression) {
                return Double(tail[range])
            }
        }
        return nil
    }

    private func firstDoubleAfterAliases(_ aliases: [String], in text: String) -> Double? {
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            let lower = String(line).lowercased()
            for alias in aliases {
                guard let range = lower.range(of: alias) else { continue }
                let tail = lower[range.upperBound...]
                let pattern = #"[-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?"#
                if let match = tail.range(of: pattern, options: .regularExpression) {
                    return Double(tail[match])
                }
            }
        }
        return nil
    }

    private func firstMatch(in text: String, pattern: String) -> Double? {
        guard let range = text.range(of: pattern, options: .regularExpression) else { return nil }
        let captured = text[range]
        let numPattern = #"[-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?"#
        if let numRange = captured.range(of: numPattern, options: .regularExpression) {
            return Double(captured[numRange])
        }
        return nil
    }

    private func isAlnum(_ char: Character?) -> Bool {
        guard let c = char else { return false }
        return c.isLetter || c.isNumber || c == "_"
    }

    private func doubleValue(_ value: Any) -> Double? {
        if let double = value as? Double { return double }
        if let int = value as? Int { return Double(int) }
        if let number = value as? NSNumber { return number.doubleValue }
        if let string = value as? String { return Double(string) }
        return nil
    }

    private func intValue(_ value: Any?) -> Int? {
        if let int = value as? Int { return int }
        if let double = value as? Double { return Int(double) }
        if let number = value as? NSNumber { return number.intValue }
        if let string = value as? String { return Int(string) }
        return nil
    }

    private func charBefore(_ needle: String, in text: String) -> Character? {
        guard let range = text.range(of: needle) else { return nil }
        guard range.lowerBound > text.startIndex else { return nil }
        let prevIndex = text.index(before: range.lowerBound)
        return text[prevIndex]
    }
}
