import SwiftUI

struct SyntheticDataView: View {
    @Bindable var store: AppStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HeaderView(title: "Synthetic Data", subtitle: "\(store.synthetic.kind.title) generator", symbol: "sparkles")

                // The long pill at the top is the **only** way to
                // start *and* cancel a synthetic-data run from this
                // page. While the run is in flight it switches to a
                // red Cancel pill so the user can stop the job from
                // the same place they launched it — the upper-right
                // toolbar pair is reserved for the training runner
                // and no longer intercepts the synthetic cancel
                // action.
                let isRunning = store.syntheticRunner.isRunning
                Button {
                    if isRunning {
                        store.syntheticRunner.stop()
                    } else {
                        Task { await store.startSynthetic() }
                    }
                } label: {
                    Label(
                        isRunning ? "Cancel Generation" : generateButtonTitle,
                        systemImage: isRunning ? "xmark.circle.fill" : "sparkles"
                    )
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(isRunning ? .red : .accentColor)
                .controlSize(.large)

                SyntheticRunConsolePill(runner: store.syntheticRunner)

                VStack(alignment: .leading, spacing: 14) {
                    SectionTitle("Dataset Type")
                    Picker("Kind", selection: $store.synthetic.kind) {
                        ForEach(SyntheticKind.allCases) { Text($0.title).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: store.synthetic.kind) { _, kind in
                        store.synthetic.applyDefaultDataset(for: kind)
                    }
                }
                .formBlock()

                SyntheticSourceSection(config: $store.synthetic)
                    .environment(store)
                SyntheticSystemPromptSection(config: $store.synthetic)
                    .environment(store)
                SyntheticOutputSection(
                    config: $store.synthetic,
                    outputRoot: store.outputRoot,
                    lastRunFolder: store.syntheticRunner.lastRunFolder,
                    outputs: store.syntheticRunOutputs,
                    onRefresh: { Task { await store.refreshLocalRunOutputs() } },
                    onSelectResume: { path in store.loadSyntheticConfig(fromGeneratedDataPath: path) }
                )
                SyntheticSamplingSection(config: $store.synthetic)
                SyntheticSplitSection(config: $store.synthetic)
            }
            .padding(24)
        }
        .frame(minWidth: 360, idealWidth: 620, maxWidth: .infinity)
        .liquidGlass(cornerRadius: 18)
        .padding(16)
        .navigationTitle("Synthetic Data")
    }

    private var generateButtonTitle: String {
        store.synthetic.resumeOutputDir.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "Generate Synthetic Dataset"
            : "Continue Synthetic Dataset"
    }
}

private struct SyntheticRunConsolePill: View {
    @Bindable var runner: PythonJobRunner

    private var recentLines: [String] {
        Array(runner.logLines.suffix(4))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Circle()
                    .fill(runner.isRunning ? .green : .secondary)
                    .frame(width: 8, height: 8)
                    .scaleEffect(runner.isRunning ? 1.12 : 0.9)

                Text(runner.isRunning ? "Live Run" : "Run Console")
                    .font(.headline)

                Text(commandSummary)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)

                Spacer()

                Button {
                    runner.clearTerminal()
                } label: {
                    Label("Clear", systemImage: "eraser")
                        .labelStyle(.iconOnly)
                }
                .buttonStyle(.borderless)
                .disabled(runner.logLines.isEmpty)
            }

            RunProgressBar(runner: runner)

            if recentLines.isEmpty {
                Text("Ready")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            } else {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(Array(recentLines.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(line.contains("[Studio]") ? .secondary : .primary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(.quaternary.opacity(0.45), lineWidth: 1)
        )
        .animation(.easeInOut(duration: 0.2), value: runner.isRunning)
        .animation(.easeInOut(duration: 0.18), value: runner.logLines.count)
    }

    private var commandSummary: String {
        runner.currentCommand.isEmpty ? "Ready" : runner.currentCommand
    }
}

private struct SyntheticSourceSection: View {
    @Binding var config: SyntheticConfig
    @Environment(AppStore.self) private var store

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle("Source")
            HFAssetPicker(
                text: $config.datasetPath,
                kind: .dataset,
                placeholder: "Prompt dataset"
            )
            Picker("Backend", selection: $config.backend) {
                ForEach(SyntheticBackend.allCases) { Text($0.title).tag($0) }
            }
            .pickerStyle(.menu)
            .onChange(of: config.backend) { _, backend in
                config.applyBackendDefaults(for: backend)
            }
            if config.kind == .sft {
                if config.backend == .mlx {
                    HFAssetPicker(
                        text: $config.model,
                        kind: .model,
                        placeholder: "Generator model"
                    )
                } else {
                    SyntheticProviderModelPicker(
                        text: $config.model,
                        backend: config.backend
                    )
                }
                ToggleRow("Include system prompt in final records", isOn: $config.includeSystemPrompt)
                ToggleRow("Use ground-truth section as generation context", isOn: $config.useGroundTruth)
            } else {
                Picker("Generate", selection: $config.dpoGenerationTarget) {
                    ForEach(SyntheticDPOGenerationTarget.allCases) { target in
                        Text(target.title).tag(target)
                    }
                }
                .pickerStyle(.segmented)
                Text(config.dpoGenerationTarget.summary)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                HFAssetPicker(
                    text: $config.baseModel,
                    kind: .model,
                    placeholder: "Base model"
                )
                if config.backend == .mlx {
                    HFAssetPicker(
                        text: $config.teacherModel,
                        kind: .model,
                        placeholder: "Teacher model"
                    )
                } else {
                    SyntheticProviderModelPicker(
                        text: $config.teacherModel,
                        backend: config.backend
                    )
                }
            }
            if config.backend != .mlx {
                TextField("Base URL", text: $config.baseURL)
                SyntheticProviderKeyField(
                    store: store,
                    backend: config.backend
                )
                if config.kind == .sft {
                    ToggleRow("Generate multi-turn conversations", isOn: $config.multiturn)
                }
                if config.kind == .sft && config.multiturn {
                    HStack {
                        NumberField("Turns", value: $config.maxTurns)
                        NumberField("Concurrent", value: $config.maxConcurrent)
                        FloatingField("Multi-turn %", value: $config.multiturnPercentile)
                    }
                    TextField("Human-role model (optional)", text: $config.humanRoleModel)
                } else {
                    NumberField("Concurrent", value: $config.maxConcurrent)
                }
                Text("In-flight API requests for \(config.backend.title). This is the parallelism knob for every non-MLX backend (OpenAI / OpenRouter / Ollama / LM Studio / oMLX / Custom) — Batch below is ignored for all of them and only affects the local MLX backend.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .formBlock()
        .animation(.easeInOut(duration: 0.2), value: config.kind)
        .animation(.easeInOut(duration: 0.2), value: config.backend)
    }
}

/// Text field + dropdown + refresh button for picking a generator
/// model from a synthetic-data provider backend (OpenAI, OpenRouter,
/// Ollama, LM Studio, oMLX, Custom).
///
/// Unlike the model picker on the Train page — which lists whatever
/// is already in the local HF cache — the providers don't have a
/// "cache" we can walk, so the picker live-scrapes the provider's
/// `GET /models` (or Ollama's `GET /api/tags`) endpoint to populate
/// the dropdown. The scrape fires automatically when the picker
/// first appears and again whenever the user switches to a
/// different backend; the refresh button next to the picker
/// triggers a manual rescrape for the rare case where a new model
/// shows up mid-session.
///
/// The user can always type a model id directly into the text
/// field — the dropdown is just a convenience. Whatever the user
/// types is what the trainer receives, scraped list or not.
private struct SyntheticProviderModelPicker: View {
    @Binding var text: String
    let backend: SyntheticBackend
    @Environment(AppStore.self) private var store

    /// Tracks whether we've already kicked off a first-time scrape
    /// for the currently-mounted picker instance. The picker gets
    /// a fresh identity per backend (`.id(backend)` on its parent
    /// container), so this is reset every time the user switches
    /// providers.
    @State private var didAutoScrape = false

    private var scrapedModels: [String] {
        store.providerModels[backend] ?? []
    }

    private var isScanning: Bool {
        store.isScanningProviderModels.contains(backend)
    }

    private var lastError: String? {
        store.providerModelError[backend]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "network")
                    .foregroundStyle(.secondary)
                TextField("Generator model", text: $text)
                    .textFieldStyle(.roundedBorder)
                Menu {
                    pickerMenuContent
                } label: {
                    HStack(spacing: 2) {
                        Image(systemName: "chevron.down")
                        Text("Pick")
                            .font(.callout)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                Button {
                    Task { await store.scrapeProviderModels(backend: backend) }
                } label: {
                    if isScanning {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: "arrow.clockwise")
                    }
                }
                .buttonStyle(.borderless)
                .help("Re-scrape \(backend.title) for the latest model list")
            }
            footer
        }
        // The picker gets a fresh identity per backend so the
        // auto-scrape `didAutoScrape` flag is automatically reset on
        // provider switch — no leftover state from the previous
        // backend can leak through.
        .id(backend)
        .task {
            // `task` fires on first appearance AND every time the
            // view's identity changes (e.g. backend switch, since
            // `.id(backend)` re-mounts the subtree). Combined with
            // the `didAutoScrape` guard, this gives us a single
            // auto-scrape per appearance, exactly the behavior we
            // want.
            guard !didAutoScrape else { return }
            didAutoScrape = true
            await store.scrapeProviderModels(backend: backend)
        }
    }

    // MARK: - Menu

    @ViewBuilder
    private var pickerMenuContent: some View {
        if isScanning && scrapedModels.isEmpty {
            Text("Fetching \(backend.title) models…")
                .foregroundStyle(.secondary)
        } else if scrapedModels.isEmpty {
            if let lastError {
                Text(lastError)
                    .foregroundStyle(.secondary)
            } else {
                Text("No models scraped yet — click the refresh button")
                    .foregroundStyle(.secondary)
            }
        } else {
            Section("\(backend.title) models (\(scrapedModels.count))") {
                ForEach(scrapedModels, id: \.self) { model in
                    Button {
                        text = model
                    } label: {
                        Label(model, systemImage: "sparkles")
                    }
                }
            }
        }
    }

    // MARK: - Footer

    @ViewBuilder
    private var footer: some View {
        HStack(spacing: 6) {
            if let lastError {
                Label(lastError, systemImage: "exclamationmark.triangle")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .lineLimit(2)
            } else if isScanning {
                Text("Fetching live model list from \(backend.title)…")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else if !scrapedModels.isEmpty {
                Text("\(scrapedModels.count) model\(scrapedModels.count == 1 ? "" : "s") scraped from \(backend.title)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else {
                Text("Click the refresh button to fetch models from \(backend.title)")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            Spacer()
        }
    }
}

/// Password-style field for the synthetic-data provider API key.
///
/// Mirrors `HuggingFaceTokenField` in the Settings view: the field
/// never holds the secret in its own state. Instead, it shows a
/// placeholder ("••• set" when a key is already in the Keychain for
/// the current provider, "Paste key…" when not) and only switches
/// into a real text editor once the user clicks "Set…" or
/// "Replace…" — at which point the new value is committed to the
/// per-provider Keychain slot on save and the draft is cleared from
/// memory.
///
/// Each provider has its own Keychain slot (see
/// `AppStore.syntheticProviderKeyIsSet`), so the saved-key indicator
/// automatically updates to reflect the *currently selected*
/// backend's state when the user switches providers in the picker
/// above. We force a re-render of the field with `.id(backend)` so a
/// pending "in edit" state for one provider is discarded the moment
/// the user picks another provider.
private struct SyntheticProviderKeyField: View {
    @Bindable var store: AppStore
    let backend: SyntheticBackend

    /// `true` while the user is in the middle of typing a new key.
    /// When `false`, the field renders the placeholder, never the
    /// real stored value.
    @State private var isEditing = false
    @State private var draft = ""

    private var isSet: Bool {
        store.syntheticProviderKeyIsSet[backend] ?? false
    }

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "key.fill")
                .foregroundStyle(.secondary)
            Group {
                if isEditing {
                    SecureField("sk-…", text: $draft)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit(commit)
                } else {
                    // The placeholder is the only thing visible when
                    // not editing. We intentionally never read the
                    // key back from the Keychain into the field —
                    // that would defeat the point of a secret field.
                    Text(placeholder)
                        .foregroundStyle(isSet ? .primary : .secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(.quaternary.opacity(0.3))
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .stroke(Color.secondary.opacity(0.25), lineWidth: 1)
                        )
                }
            }
            if isEditing {
                Button("Save") { commit() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Button("Cancel") {
                    draft = ""
                    isEditing = false
                }
            } else {
                Button(isSet ? "Replace…" : "Set…") {
                    draft = ""
                    isEditing = true
                }
                if isSet {
                    Button {
                        store.clearSyntheticProviderKey(for: backend)
                    } label: {
                        Image(systemName: "trash")
                    }
                    .help("Remove the stored key for \(backend.title)")
                }
            }
        }
        // Discard a pending "in edit" state and draft the moment the
        // user picks a different provider — otherwise typing a key
        // for OpenRouter, then switching to OpenAI, would briefly
        // show the OpenRouter draft in the OpenAI field.
        .id(backend)
        .onChange(of: backend) { _, _ in
            draft = ""
            isEditing = false
        }
    }

    private var placeholder: String {
        isSet
            ? "••• \(backend.title) key stored in Keychain"
            : "No key set — optional for \(backend.title)"
    }

    private func commit() {
        let trimmed = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        store.setSyntheticProviderKey(trimmed, for: backend)
        draft = ""
        isEditing = false
    }
}

private struct SyntheticSystemPromptSection: View {
    @Binding var config: SyntheticConfig

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle("System Prompt")
            SystemPromptPicker(
                text: $config.systemPrompt,
                placeholder: "System prompt text or file"
            )
        }
        .formBlock()
    }
}

private struct SyntheticOutputSection: View {
    @Binding var config: SyntheticConfig
    let outputRoot: String
    let lastRunFolder: String
    let outputs: [LocalRunOutput]
    let onRefresh: () -> Void
    let onSelectResume: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle("Output")
            TextField("Run folder name", text: $config.runFolderName)
            SyntheticResumeRunPicker(
                selection: $config.resumeOutputDir,
                outputs: outputs,
                onRefresh: onRefresh,
                onSelectResume: onSelectResume
            )
            HStack(alignment: .top, spacing: 12) {
                InfoPill(
                    text: runFolderDisplayName,
                    symbol: "folder",
                    openPath: preferredRunFolderPath
                )
                InfoPill(text: "Root: \(outputRoot)", symbol: "externaldrive", openPath: outputRoot)
            }
            Text("Generated datasets are saved under the run folder's generated-data directory. To continue a partial run, select the existing run here and keep Samples set to the desired final total.")
                .foregroundStyle(.secondary)
                .font(.callout)
            if !lastRunFolder.isEmpty {
                Text(lastRunFolder)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .formBlock()
    }

    private var runFolderDisplayName: String {
        if !config.resumeOutputDir.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return URL(fileURLWithPath: config.resumeOutputDir).deletingLastPathComponent().lastPathComponent
        }
        return config.runFolderName.isEmpty ? config.automaticRunFolderName() : config.resolvedRunFolderName()
    }

    private var preferredRunFolderPath: String {
        let resumeOutput = config.resumeOutputDir.trimmingCharacters(in: .whitespacesAndNewlines)
        if !resumeOutput.isEmpty {
            return URL(fileURLWithPath: NSString(string: resumeOutput).expandingTildeInPath, isDirectory: true)
                .deletingLastPathComponent()
                .path
        }
        if !lastRunFolder.isEmpty {
            return lastRunFolder
        }
        let expandedRoot = NSString(string: outputRoot).expandingTildeInPath
        return URL(fileURLWithPath: expandedRoot, isDirectory: true)
            .appending(path: runFolderDisplayName, directoryHint: .isDirectory)
            .path
    }
}

private struct SyntheticResumeRunPicker: View {
    @Binding var selection: String
    let outputs: [LocalRunOutput]
    let onRefresh: () -> Void
    let onSelectResume: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Picker("Continue run", selection: $selection) {
                    Text("Start fresh").tag("")
                    ForEach(outputs) { output in
                        Text(output.name).tag(output.path)
                    }
                    if !selection.isEmpty && !outputs.contains(where: { $0.path == selection }) {
                        Text("Custom: \(URL(fileURLWithPath: selection).deletingLastPathComponent().lastPathComponent)")
                            .tag(selection)
                    }
                }
                .pickerStyle(.menu)
                .onChange(of: selection) { _, path in
                    guard !path.isEmpty else { return }
                    onSelectResume(path)
                }

                Button {
                    onRefresh()
                } label: {
                    Label("Refresh runs", systemImage: "arrow.clockwise")
                }
                .labelStyle(.iconOnly)
                .help("Refresh synthetic runs")
            }

            if outputs.isEmpty {
                Text("No generated-data folders found under the output root yet.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if !selection.isEmpty {
                Text(selection)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
            }
        }
    }
}

private struct SyntheticSamplingSection: View {
    @Binding var config: SyntheticConfig

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle("Sampling")
            HStack {
                NumberField("Samples", value: $config.numSamples)
                NumberField("Batch", value: $config.batchSize)
                NumberField("Seed", value: $config.seed)
            }
            // `batchSize` only affects the local MLX backend —
            // it's the number of examples the local generator
            // processes in parallel per step. Cloud / server
            // providers (OpenAI / OpenRouter / Ollama / LM Studio /
            // oMLX / Custom) ignore it and use the `Concurrent`
            // field in the Source section above as their
            // in-flight request count. We surface a short
            // explainer so the user doesn't have to guess which
            // knob is the live one for the currently selected
            // backend.
            Text(batchExplainer)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            ToggleRow("Use custom generation settings", isOn: $config.useGenerationSettings)
            if config.useGenerationSettings {
                HStack {
                    NumberField("Tokens", value: $config.maxTokens)
                    FloatingField("Temp", value: $config.temperature)
                    FloatingField("Top P", value: $config.topP)
                    FloatingField("Min P", value: $config.minP)
                }
                HStack {
                    NumberField("Top K", value: $config.topK)
                    NumberField("Min Keep", value: $config.minTokensToKeep)
                    FloatingField("XTC Prob", value: $config.xtcProbability)
                    FloatingField("XTC Threshold", value: $config.xtcThreshold)
                }
            } else if config.backend != .mlx {
                Text("API requests will omit generation settings and let the provider or local server choose its defaults.")
                    .foregroundStyle(.secondary)
                    .font(.callout)
            }
        }
        .formBlock()
        .animation(.easeInOut(duration: 0.2), value: config.useGenerationSettings)
    }

    /// Short explainer under the Samples / Batch / Seed row that
    /// tells the user which of the two parallelism knobs (Batch
    /// here vs Concurrent in the Source section) is actually
    /// driving the run for the currently selected backend.
    private var batchExplainer: String {
        if config.backend == .mlx {
            return "Batch is the parallelism knob for the local MLX generator. The Concurrent field in Source is hidden for MLX and only appears for the non-MLX backends below."
        }
        return "Batch is ignored for \(config.backend.title) — and for every other non-MLX backend too (OpenAI / OpenRouter / Ollama / LM Studio / oMLX / Custom all use the same OpenAI-compatible path). Batch only affects the local MLX backend. Use the Concurrent field in Source above to control parallelism for this provider."
    }
}

private struct SyntheticSplitSection: View {
    @Binding var config: SyntheticConfig

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle("Splits")
            HStack {
                TextField("Validation split", text: $config.validSplit)
                TextField("Test split", text: $config.testSplit)
            }
            Text("Leave split fields empty to write all generated examples to the train parquet.")
                .foregroundStyle(.secondary)
                .font(.callout)
        }
        .formBlock()
    }
}
