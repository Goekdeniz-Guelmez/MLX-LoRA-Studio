import Foundation

/// Discovers previous training runs from the output root on disk. Each
/// run is a folder written by `PythonJobRunner.makeRunFolder` that
/// contains a `run_spec.json` (the `TrainingConfig` payload) and,
/// since the metrics-persistence change, a `metrics.json` file.
///
/// The scanner is intentionally tolerant:
/// - A folder without a recognisable spec is skipped (it was never a
///   training run — e.g. a synthetic-data folder).
/// - A folder with a `run_spec.json` we can't decode still shows up on
///   the Runs page (so the user can find it in Finder), but with
///   `spec == nil` and an empty metrics array.
/// - The scanner is nonisolated so callers can hop off the main actor
///   for the directory walk (one `contentsOfDirectory` per subdir, no
///   async filesystem APIs needed).
enum RunArchive {
    enum DeleteError: LocalizedError {
        case outsideOutputRoot

        var errorDescription: String? {
            switch self {
            case .outsideOutputRoot:
                "That run folder is outside the configured output folder, so it was not deleted."
            }
        }
    }

    /// Walk the output root, decode every Studio run, and return a
    /// newest-first list. Training folders get metrics and training
    /// settings; synthetic folders get their dataset-generation config
    /// plus a small JSONL preview.
    static func discoverPersistedRuns(outputRoot: String) -> [PersistedRun] {
        let expandedRoot = NSString(string: outputRoot).expandingTildeInPath
        let rootURL = URL(fileURLWithPath: expandedRoot, isDirectory: true)
        guard let children = try? FileManager.default.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: [.contentModificationDateKey, .isDirectoryKey, .creationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        let sorted = children.sorted { lhs, rhs in
            let leftDate = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let rightDate = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return leftDate > rightDate
        }

        return sorted.compactMap { runURL in
            guard ((try? runURL.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false) else {
                return nil
            }
            return run(at: runURL)
        }
    }

    static func deletePersistedRun(_ run: PersistedRun, outputRoot: String) throws {
        let expandedRoot = NSString(string: outputRoot).expandingTildeInPath
        let rootURL = URL(fileURLWithPath: expandedRoot, isDirectory: true).standardizedFileURL
        let runURL = run.folderURL.standardizedFileURL

        guard runURL.path.hasPrefix(rootURL.path + "/") else {
            throw DeleteError.outsideOutputRoot
        }

        try FileManager.default.removeItem(at: runURL)
    }

    static func resumeCandidate(for run: PersistedRun) -> TrainingResumeCandidate? {
        guard run.kind == .training, run.spec != nil else { return nil }
        let adaptersURL = run.folderURL.appending(path: "adapters", directoryHint: .isDirectory)
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: adaptersURL,
            includingPropertiesForKeys: [.contentModificationDateKey, .isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        let candidates = files.compactMap { url -> (url: URL, step: Int?, modified: Date)? in
            guard ((try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) ?? false) else {
                return nil
            }
            let filename = url.lastPathComponent
            guard filename == "adapters.safetensors" || filename.hasSuffix("_adapters.safetensors") else {
                return nil
            }
            let modified = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                ?? .distantPast
            return (url, checkpointStep(from: filename), modified)
        }

        guard let best = candidates.max(by: { lhs, rhs in
            switch (lhs.step, rhs.step) {
            case let (l?, r?) where l != r:
                return l < r
            case (_?, nil):
                return false
            case (nil, _?):
                return true
            default:
                return lhs.modified < rhs.modified
            }
        }) else {
            return nil
        }
        return TrainingResumeCandidate(adapterFile: best.url, step: best.step)
    }

    static func continuationCandidate(for run: PersistedRun) -> TrainingResumeCandidate? {
        guard let spec = run.spec,
              spec.epochs == 0,
              let candidate = resumeCandidate(for: run),
              let step = candidate.step,
              spec.iters > step else {
            return nil
        }
        return candidate
    }

    // MARK: - Per-folder builder

    /// Build a `PersistedRun` for a single folder. Returns `nil` if the
    /// folder is clearly not a Studio run (no spec file at all).
    private static func run(at folderURL: URL) -> PersistedRun? {
        let name = folderURL.lastPathComponent
        let createdAt = (try? folderURL.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
            ?? Date.distantPast

        // Detect the kind by which spec file is present. Training
        // folders always have `run_spec.json`; synthetic folders have
        // `synthetic_spec.json` and usually a generated-data JSONL file
        // we can preview without loading the whole dataset.
        let trainingSpec = folderURL.appending(path: "run_spec.json")
        let syntheticSpec = folderURL.appending(path: "synthetic_spec.json")
        let hfUploadSpec = folderURL.appending(path: "hf_upload_spec.json")

        let kind: PersistedRun.Kind
        let spec: TrainingConfig?
        let decodedSyntheticSpec: SyntheticConfig?
        let syntheticSamples: [SyntheticDatasetSample]
        let command: String

        if FileManager.default.fileExists(atPath: trainingSpec.path) {
            kind = .training
            let decoded = TrainingConfig.decoded(from: trainingSpec)
            spec = decoded
            decodedSyntheticSpec = nil
            syntheticSamples = []
            command = decoded?.reconstructedCommand ?? "python -m mlx_lm_lora train"
        } else if FileManager.default.fileExists(atPath: hfUploadSpec.path) {
            kind = .hfUpload
            spec = nil
            decodedSyntheticSpec = nil
            syntheticSamples = []
            command = "huggingface_hub upload …"
        } else if FileManager.default.fileExists(atPath: syntheticSpec.path) {
            kind = .synthetic
            spec = nil
            decodedSyntheticSpec = SyntheticConfig.decoded(from: syntheticSpec)
            syntheticSamples = readSyntheticSamples(in: folderURL)
            command = "python -m mlx_lm_lora synthetic_data …"
        } else {
            // Not a Studio run folder — bail.
            return nil
        }

        let metricsURL = folderURL.appending(path: TrainingMetricIO.filename)
        let metrics = TrainingMetricIO.read(from: metricsURL)

        let title: String
        if let spec {
            let modelName = spec.model
                .split(separator: "/")
                .last
                .map(String.init) ?? spec.model
            title = "\(spec.trainMode.title) · \(spec.trainType.title) · \(modelName)"
        } else {
            // Fall back to a humanised folder name for runs we can't
            // decode fully.
            title = humaniseFolderName(name, kind: kind)
        }

        return PersistedRun(
            id: name,
            folderURL: folderURL,
            kind: kind,
            createdAt: createdAt,
            title: title,
            spec: spec,
            metrics: metrics,
            syntheticSpec: decodedSyntheticSpec,
            syntheticSamples: syntheticSamples,
            command: command
        )
    }

    private static func readSyntheticSamples(in folderURL: URL, limit: Int = 3) -> [SyntheticDatasetSample] {
        let candidateFiles = [
            folderURL.appending(path: "generated-data/output_full.jsonl"),
            folderURL.appending(path: "generated-data/train.jsonl"),
            folderURL.appending(path: "generated-data/data/train.jsonl"),
            folderURL.appending(path: "output_full.jsonl"),
            folderURL.appending(path: "train.jsonl"),
            folderURL.appending(path: "data/train.jsonl")
        ]

        for fileURL in candidateFiles where FileManager.default.fileExists(atPath: fileURL.path) {
            let samples = readJSONLSamples(from: fileURL, limit: limit)
            if !samples.isEmpty {
                return samples
            }
        }
        return []
    }

    private static func readJSONLSamples(from fileURL: URL, limit: Int) -> [SyntheticDatasetSample] {
        guard limit > 0 else { return [] }

        var samples: [SyntheticDatasetSample] = []
        for (offset, line) in readFirstLines(from: fileURL, limit: limit * 8).enumerated() {
            guard samples.count < limit else { break }
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, let data = trimmed.data(using: .utf8) else { continue }
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            let fields = sampleFields(from: object ?? ["text": trimmed])
            guard !fields.isEmpty else { continue }
            samples.append(
                SyntheticDatasetSample(
                    index: offset + 1,
                    sourceFile: fileURL.lastPathComponent,
                    fields: fields
                )
            )
        }
        return samples
    }

    private static func readFirstLines(from fileURL: URL, limit: Int) -> [String] {
        guard limit > 0, let handle = try? FileHandle(forReadingFrom: fileURL) else {
            return []
        }
        defer { try? handle.close() }

        var lines: [String] = []
        var buffer = Data()
        let newline = Data([0x0A])

        while lines.count < limit {
            guard let chunk = try? handle.read(upToCount: 8192), !chunk.isEmpty else {
                break
            }
            buffer.append(chunk)
            while lines.count < limit, let range = buffer.firstRange(of: newline) {
                let lineData = buffer[..<range.lowerBound]
                if let line = String(data: lineData, encoding: .utf8) {
                    lines.append(line)
                }
                buffer.removeSubrange(..<range.upperBound)
            }
        }

        if lines.count < limit, !buffer.isEmpty, let line = String(data: buffer, encoding: .utf8) {
            lines.append(line)
        }
        return lines
    }

    private static func sampleFields(from object: [String: Any]) -> [SyntheticSampleField] {
        let priority = [
            "prompt", "completion", "chosen", "rejected", "messages",
            "question", "answer", "text", "system", "source"
        ]
        var fields: [SyntheticSampleField] = []
        var seen = Set<String>()

        for key in priority {
            if let value = object[key] {
                appendSampleField(key, value, to: &fields, seen: &seen)
            }
        }
        for key in object.keys.sorted() where !seen.contains(key) {
            appendSampleField(key, object[key] as Any, to: &fields, seen: &seen)
        }
        return fields
    }

    private static func appendSampleField(
        _ key: String,
        _ value: Any,
        to fields: inout [SyntheticSampleField],
        seen: inout Set<String>
    ) {
        let text = displayString(for: value)
        guard !text.isEmpty else { return }
        seen.insert(key)
        fields.append(SyntheticSampleField(key: key, value: text))
    }

    private static func displayString(for value: Any) -> String {
        switch value {
        case let string as String:
            return string.trimmingCharacters(in: .whitespacesAndNewlines)
        case let array as [[String: Any]]:
            return array.enumerated().map { index, message in
                let role = (message["role"] as? String) ?? "message \(index + 1)"
                let content = message["content"].map(displayString(for:)) ?? ""
                return "\(role): \(content)"
            }
            .joined(separator: "\n\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        case let array as [Any]:
            return array.map(displayString(for:))
                .filter { !$0.isEmpty }
                .joined(separator: "\n")
        case let dictionary as [String: Any]:
            if let data = try? JSONSerialization.data(withJSONObject: dictionary, options: [.prettyPrinted, .sortedKeys]),
               let text = String(data: data, encoding: .utf8) {
                return text
            }
            return ""
        case let number as NSNumber:
            return number.stringValue
        case is NSNull:
            return ""
        default:
            return "\(value)"
        }
    }

    private static func checkpointStep(from filename: String) -> Int? {
        guard filename.hasSuffix("_adapters.safetensors") else { return nil }
        let prefix = filename.replacingOccurrences(of: "_adapters.safetensors", with: "")
        return Int(prefix)
    }

    /// `synthetic-sft-ultrafeedback-prompts-flat-rlhf-2026-05-19-124511` →
    /// `Synthetic SFT · 2026-05-19 12:45`
    private static func humaniseFolderName(_ name: String, kind: PersistedRun.Kind) -> String {
        let prefix: String
        switch kind {
        case .training: prefix = "Training"
        case .synthetic: prefix = "Synthetic"
        case .hfUpload: prefix = "HF Upload"
        }
        return "\(prefix) · \(name)"
    }
}
