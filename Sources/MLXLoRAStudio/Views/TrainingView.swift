import SwiftUI
import UniformTypeIdentifiers

struct TrainingView: View {
    @Bindable var store: AppStore
    // When the window gets narrow, the user can collapse the Run Console
    // to a thin strip and bring it back later with a single click. We
    // also auto-collapse it if the available width drops below a
    // comfortable threshold (see the GeometryReader below).
    @State private var consoleCollapsed: Bool = false
    @State private var manualConsoleCollapsed: Bool? = nil

    var body: some View {
        // HSplitView (backed by NSSplitView) gives us a real draggable
        // divider with no overlap. Each side uses `minWidth` as a hard
        // floor and `idealWidth` as the starting size; the `Layout`
        // priorities below make sure the form (left) wins when the
        // window is too small to give both columns their ideal width.
        GeometryReader { proxy in
            let available = proxy.size.width
            // Threshold where the two columns together can't both be
            // at their ideal size. Below this, we auto-collapse the
            // console unless the user has manually toggled it.
            let shouldAutoCollapse = available < 900
            let isCollapsed = effectiveCollapsed(shouldAutoCollapse: shouldAutoCollapse)
            let settingsWidth = available * 0.6
            let consoleWidth = available - settingsWidth

            HSplitView {
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        HeaderView(
                            title: "Train",
                            subtitle: "\(store.training.trainMode.title) using \(store.training.trainType.title)",
                            symbol: "cpu"
                        )

                        ModeStrip(config: $store.training)
                        ModelDataSection(config: $store.training)
                            .environment(store)
                        OutputSection(config: $store.training, outputRoot: store.outputRoot, lastRunFolder: store.trainingRunner.lastRunFolder)
                        CoreTrainingSection(config: $store.training)
                        PreferenceSection(config: $store.training)
                        GRPOSection(config: $store.training)
                        OnlinePreferenceSection(config: $store.training)
                        DatasetMappingSection(config: $store.training)
                        QATSection(config: $store.training)
                    }
                    .padding(24)
                }
                .layoutPriority(1)
                .frame(
                    minWidth: 360,
                    idealWidth: isCollapsed ? 620 : settingsWidth,
                    maxWidth: isCollapsed ? .infinity : settingsWidth
                )
                .trainSettingsPane(cornerRadius: 18)
                .padding(16)

                LiveRunPanel(
                    runner: store.trainingRunner,
                    collapsed: Binding(
                        get: { isCollapsed },
                        set: { manualConsoleCollapsed = $0 }
                    )
                )
                .frame(
                    minWidth: isCollapsed ? 56 : 260,
                    idealWidth: isCollapsed ? 56 : consoleWidth,
                    maxWidth: isCollapsed ? 56 : consoleWidth
                )
            }
        }
        .navigationTitle("Train")
    }

    /// Resolves the final collapse state, taking both the manual
    /// user toggle and the auto-collapse rule into account. A manual
    /// override always wins; if the user hasn't touched it, the
    /// console hides itself below 900pt so the form stays readable.
    private func effectiveCollapsed(shouldAutoCollapse: Bool) -> Bool {
        if let manual = manualConsoleCollapsed { return manual }
        return shouldAutoCollapse
    }
}

private struct ModeStrip: View {
    @Binding var config: TrainingConfig

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionTitle("Algorithm")
            Picker("Model family", selection: $config.modelFamily) {
                ForEach(ModelFamily.allCases) { family in
                    Text(family.title).tag(family)
                }
            }
            .pickerStyle(.segmented)

            if config.modelFamily == .visionLanguage {
                Text("VLM mode fine-tunes the language model only, then exports a complete vision-language model by merging the trained text weights back into the original VLM.")
                    .foregroundStyle(.secondary)
                    .font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // The 8-mode segmented picker (SFT/DPO/CPO/ORPO/GRPO/Online
            // DPO/XPO/RLHF Reinforce/PPO). We use the original
            // segmented style — the wide rows make it easy to scan all
            // available algorithms at a glance, and a dropdown menu
            // hides the choice set behind a click. The user explicitly
            // asked to keep the segmented control, so the picker now
            // wraps in a `ScrollView(.horizontal)` on narrow windows
            // instead of falling back to a menu — at worst the row
            // scrolls, but the user always sees all the labels.
            ScrollView(.horizontal, showsIndicators: false) {
                Picker("Mode", selection: $config.trainMode) {
                    ForEach(TrainMode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            .onChange(of: config.trainMode) { _, mode in
                config.applyDefaultDataset(for: mode)
            }

            HStack(alignment: .top, spacing: 12) {
                InfoPill(text: config.trainMode.family, symbol: "tag")
                InfoPill(text: config.trainMode.datasetHint, symbol: "tray")
            }

            Text(config.trainMode.summary)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            // The Fine-tune picker sits right under the algorithm
            // list so the user can see all three training shapes
            // (LoRA, DoRA, full) at once and pick one. Same
            // horizontal-scrolling segmented style as the algorithm
            // picker, for the same reasons.
            SectionTitle("Fine-tune")
            ScrollView(.horizontal, showsIndicators: false) {
                Picker("Fine-tune", selection: $config.trainType) {
                    ForEach(TrainType.allCases) { type in
                        Text(type.title).tag(type)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }

            // LoRA / DoRA configuration. The "adapter" controls
            // (rank, scale, dropout) and the layer count are only
            // meaningful for parameter-efficient methods — full
            // fine-tuning trains every weight, so the same knobs
            // would be dead UI. We render them only when the user
            // picks LoRA or DoRA, and they animate in/out as the
            // fine-tune selection changes so the card's height
            // stays honest.
            if config.trainType == .lora || config.trainType == .dora {
                VStack(alignment: .leading, spacing: 10) {
                    SectionTitle("LoRA Settings")
                    HStack {
                        NumberField("Layers", value: $config.numLayers)
                        NumberField("Rank", value: $config.rank)
                    }
                    HStack {
                        FloatingField("Scale", value: $config.scale)
                        FloatingField("Dropout", value: $config.dropout)
                    }
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }

            // Quantization is always visible but as a dropdown
            // menu — it has 5 cases and the user only ever changes
            // it once per run, so the segmented-list treatment that
            // fits algorithms + fine-tune doesn't add value here.
            Picker("Quantization", selection: $config.quantization) {
                ForEach(Quantization.allCases) { quant in
                    Text(quant.title).tag(quant)
                }
            }
        }
        .formBlock()
        .animation(.easeInOut(duration: 0.2), value: config.trainType)
        .animation(.easeInOut(duration: 0.2), value: config.trainMode)
        .animation(.easeInOut(duration: 0.2), value: config.modelFamily)
        .animation(.easeInOut(duration: 0.2), value: config.test)
    }
}

private struct ModelDataSection: View {
    @Binding var config: TrainingConfig
    @Environment(AppStore.self) private var store

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle("Model And Data")
            HFAssetPicker(
                text: $config.model,
                kind: .model,
                placeholder: config.modelFamily == .visionLanguage ? "Trainable text model or HF repo" : "Model or Hugging Face repo",
                footer: cacheFooter
            )
            if config.modelFamily == .visionLanguage {
                HFAssetPicker(
                    text: $config.vlmModel,
                    kind: .model,
                    placeholder: "Original VLM repo or local folder",
                    footer: "The vision tower, processor, and VLM config are preserved from this model."
                )
                Text("Use a text model that matches the VLM's language model. After training, Studio writes a full MLX-VLM export.")
                    .foregroundStyle(.secondary)
                    .font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HFAssetPicker(
                text: $config.data,
                kind: .dataset,
                placeholder: "Dataset path or local data folder",
                footer: nil
            )
            TextField("LM Studio export name", text: $config.lmStudioName)
            // Fine-tune (LoRA / DoRA / full) and Quantization moved
            // into the Algorithm card directly under the algorithm
            // picker, so they sit next to the controls they modify.
        }
        .formBlock()
        .animation(.easeInOut(duration: 0.2), value: config.trainMode)
        .animation(.easeInOut(duration: 0.2), value: config.modelFamily)
    }

    /// Footer line that summarises what the dropdown is showing, so the
    /// user knows whether the cache has been scanned yet.
    private var cacheFooter: String? {
        if store.isScanningHFCache { return "Scanning local HF cache…" }
        let n = store.cachedModels.count
        let d = store.cachedDatasets.count
        let s = store.syntheticRunOutputs.count
        return "Local HF cache: \(n) model\(n == 1 ? "" : "s") · \(d) dataset\(d == 1 ? "" : "s") · Synthetic runs: \(s)"
    }
}

private struct OutputSection: View {
    @Binding var config: TrainingConfig
    let outputRoot: String
    let lastRunFolder: String

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle("Output")
            TextField("Run folder name", text: $config.runFolderName)
            HStack(alignment: .top, spacing: 12) {
                InfoPill(
                    text: runFolderDisplayName,
                    symbol: "folder",
                    openPath: lastRunFolder.isEmpty ? plannedRunFolderPath : lastRunFolder
                )
                InfoPill(text: "Root: \(outputRoot)", symbol: "externaldrive", openPath: outputRoot)
            }
            Text("Weights are saved to the run folder's adapters directory. Set the parent output folder in Settings.")
                .foregroundStyle(.secondary)
                .font(.callout)
            if config.modelFamily == .visionLanguage {
                TextField("Full VLM export folder (optional)", text: $config.vlmOutputPath)
                ToggleRow("Dequantize text weights for VLM export", isOn: $config.vlmDequantize)
                Text("Leave the export folder empty to save the merged VLM inside the run folder as full-vlm.")
                    .foregroundStyle(.secondary)
                    .font(.callout)
            }
            if !lastRunFolder.isEmpty {
                Text(lastRunFolder)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .formBlock()
        .animation(.easeInOut(duration: 0.2), value: config.qatEnable)
        .animation(.easeInOut(duration: 0.2), value: config.modelFamily)
    }

    private var runFolderDisplayName: String {
        config.runFolderName.isEmpty ? config.automaticRunFolderName() : config.resolvedRunFolderName()
    }

    private var plannedRunFolderPath: String {
        let expandedRoot = NSString(string: outputRoot).expandingTildeInPath
        return URL(fileURLWithPath: expandedRoot, isDirectory: true)
            .appending(path: runFolderDisplayName, directoryHint: .isDirectory)
            .path
    }
}

private struct CoreTrainingSection: View {
    @Binding var config: TrainingConfig

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle("Training Settings")
            HStack {
                NumberField("Iterations", value: $config.iters)
                NumberField("Epochs", value: $config.epochs)
                NumberField("Batch", value: $config.batchSize)
            }
            HStack {
                NumberField("Max Seq", value: $config.maxSeqLength)
                NumberField("Seed", value: $config.seed)
            }
            HStack {
                FloatingField("Learning Rate", value: $config.learningRate)
                Picker("Optimizer", selection: $config.optimizer) {
                    ForEach(OptimizerKind.allCases) { Text($0.title).tag($0) }
                }
            }
            Picker("LR Schedule", selection: $config.learningRateSchedule) {
                ForEach(LearningRateScheduleKind.allCases) { Text($0.title).tag($0) }
            }
            if config.learningRateSchedule == .cosineDecay {
                HStack {
                    NumberField("Warmup", value: $config.lrWarmupSteps)
                    FloatingField("Warmup Init", value: $config.lrWarmupInit)
                    FloatingField("Decay Fraction", value: $config.lrDecayFraction)
                    FloatingField("Final LR", value: $config.lrFinal)
                }
                Text("Cosine decay uses the learning rate above as the peak, then decays for the selected fraction of total iterations.")
                    .foregroundStyle(.secondary)
                    .font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                NumberField("Report", value: $config.stepsPerReport)
                NumberField("Eval", value: $config.stepsPerEval)
                NumberField("Save", value: $config.saveEvery)
                NumberField("Val Batches", value: $config.valBatches)
            }
            ToggleRow("Gradient checkpointing", isOn: $config.gradCheckpoint)
            if config.gradCheckpoint {
                NumberField("Gradient accumulation steps", value: $config.gradientAccumulationSteps)
            }
            ToggleRow("Efficient long context", isOn: $config.efficientLongContext)
            if config.efficientLongContext {
                NumberField("Sequence step size", value: $config.seqStepSize)
            }
            ToggleRow("Mask prompt loss", isOn: $config.maskPrompt)
            ToggleRow(config.modelFamily == .visionLanguage ? "Export full VLM after training" : "Fuse merged model after training", isOn: $config.fuse)
            if config.fuse, config.modelFamily == .text {
                ToggleRow("Dequantize merged model", isOn: $config.fuseDequantize)
                ToggleRow("Remove adapters after fuse", isOn: $config.fuseRemoveAdapters)
            }
            ToggleRow("Evaluate test split", isOn: $config.test)
            if config.test {
                NumberField("Test Batches", value: $config.testBatches)
            }
        }
        .formBlock()
        .animation(.easeInOut(duration: 0.2), value: config.gradCheckpoint)
        .animation(.easeInOut(duration: 0.2), value: config.efficientLongContext)
        .animation(.easeInOut(duration: 0.2), value: config.learningRateSchedule)
        .animation(.easeInOut(duration: 0.2), value: config.fuse)
        .animation(.easeInOut(duration: 0.2), value: config.modelFamily)
    }
}

private struct PreferenceSection: View {
    @Binding var config: TrainingConfig

    var body: some View {
        // SFT never reads the preference/judge knobs (it has no reward
        // signal and no reference model), and the online-preference
        // modes (online DPO, XPO, PPO, REINFORCE) get their own
        // `OnlinePreferenceSection` with the per-mode temperature /
        // judge-system-prompt / epsilon fields. Hide the whole block
        // for those modes so the trainer view stays free of dead
        // controls; the offline preference algorithms (DPO / CPO /
        // ORPO) and GRPO still use Beta / Delta / Loss below.
        let mode = config.trainMode
        if mode != .sft
            && mode != .onlineDPO
            && mode != .xpo
            && mode != .ppo
            && mode != .rlhfReinforce
        {
            VStack(alignment: .leading, spacing: 14) {
                SectionTitle("Preference And Judge")
                HStack {
                    FloatingField("Beta", value: $config.beta)
                    FloatingField("Delta", value: $config.delta)
                    FloatingField("Reward Scale", value: $config.rewardScaling)
                }
                Picker("Loss", selection: $config.dpoCpoLossType) {
                    Text("Sigmoid").tag("sigmoid")
                    Text("Hinge").tag("hinge")
                    Text("IPO").tag("ipo")
                    Text("DPOP").tag("dpop")
                }
                if config.trainMode.needsReference {
                    TextField("Reference model path", text: $config.referenceModelPath)
                }
                if config.trainMode.needsJudge {
                    TextField("Judge model or human", text: $config.judge)
                }
            }
            .formBlock()
        }
    }
}

private struct GRPOSection: View {
    @Binding var config: TrainingConfig
    @State private var showingRewardFileImporter = false

    var body: some View {
        if config.trainMode == .grpo {
            VStack(alignment: .leading, spacing: 14) {
                SectionTitle("GRPO Generation And Rewards")
                HStack {
                    NumberField("Group", value: $config.groupSize)
                    NumberField("Completion", value: $config.maxCompletionLength)
                    FloatingField("Temp", value: $config.temperature)
                    FloatingField("Epsilon", value: $config.epsilon)
                }
                HStack {
                    FloatingField("Top P", value: $config.topP)
                    NumberField("Top K", value: $config.topK)
                    FloatingField("Min P", value: $config.minP)
                    TextField("Epsilon high", text: $config.epsilonHigh)
                }
                HStack {
                    Picker("GRPO Loss", selection: $config.grpoLossType) {
                        Text("GRPO").tag("grpo")
                        Text("BNPO").tag("bnpo")
                        Text("DR GRPO").tag("dr_grpo")
                    }
                    Picker("Importance", selection: $config.importanceSamplingLevel) {
                        Text("Default").tag("")
                        Text("Token").tag("token")
                        Text("Sequence").tag("sequence")
                    }
                }
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        SectionTitle("Default Reward Functions")
                        Spacer()
                        Button("Use All Defaults") {
                            config.useDefaultGRPORewardFunctions()
                        }
                        .buttonStyle(.borderless)
                        .help("Clear custom selection so GRPO uses the backend default reward set.")
                    }
                    ForEach(GRPORewardFunction.defaults) { reward in
                        GRPORewardFunctionRow(
                            reward: reward,
                            isSelected: config.selectedGRPORewardFunctionNames.contains(reward.name)
                        ) {
                            config.toggleDefaultGRPORewardFunction(reward.name)
                        }
                    }
                    Text("Leave the custom list empty to use all defaults. Selecting rows writes the function names that will be passed to the trainer.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                TextField("Custom reward function names, comma-separated", text: $config.rewardFunctions)
                    .font(.system(.body, design: .monospaced))
                TextField("Reward weights, e.g. [2.0, 0.5, 0.5, 0.5, 0.5]", text: $config.rewardWeights)
                    .font(.system(.body, design: .monospaced))
                HStack(alignment: .top, spacing: 10) {
                    TextField("Reward functions Python file", text: $config.rewardFunctionsFile)
                        .font(.system(.body, design: .monospaced))
                    Button {
                        showingRewardFileImporter = true
                    } label: {
                        Label("Import", systemImage: "doc.badge.plus")
                    }
                }
                Text("Custom Python files should register functions with @register_reward_function(), then list their names above.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .formBlock()
            .fileImporter(
                isPresented: $showingRewardFileImporter,
                allowedContentTypes: [.pythonScript, .plainText],
                allowsMultipleSelection: false
            ) { result in
                guard case .success(let urls) = result, let url = urls.first else { return }
                config.rewardFunctionsFile = url.path
            }
        }
    }
}

private struct GRPORewardFunctionRow: View {
    let reward: GRPORewardFunction
    let isSelected: Bool
    let onToggle: () -> Void

    var body: some View {
        Button(action: onToggle) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(iconStyle)
                    .font(.title3)
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 8) {
                        Text(reward.title)
                            .font(.callout.weight(.semibold))
                        Text(reward.name)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    Text(reward.summary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .liquidGlass(cornerRadius: 8, interactive: true)
        }
        .buttonStyle(.plain)
    }

    private var iconStyle: AnyShapeStyle {
        isSelected ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary)
    }
}

private struct OnlinePreferenceSection: View {
    @Binding var config: TrainingConfig

    var body: some View {
        if config.trainMode == .onlineDPO || config.trainMode == .xpo || config.trainMode == .ppo || config.trainMode == .rlhfReinforce {
            VStack(alignment: .leading, spacing: 14) {
                SectionTitle("Online Preference")
                HStack {
                    NumberField("Completion", value: $config.maxCompletionLength)
                    if config.trainMode == .onlineDPO || config.trainMode == .xpo || config.trainMode == .ppo {
                        FloatingField("Temp", value: $config.temperature)
                    }
                    if config.trainMode == .ppo {
                        FloatingField("Epsilon", value: $config.epsilon)
                    }
                }
                Picker("Judge", selection: $config.judgeKind) {
                    ForEach(JudgeKind.allCases) { kind in
                        Text(kind.title).tag(kind)
                    }
                }
                .pickerStyle(.segmented)
                .onChange(of: config.judgeKind) { _, kind in
                    applyJudgeKind(kind)
                }
                if config.judgeKind == .llm {
                    TextField("Judge model", text: $config.judge)
                }
                HStack(alignment: .top, spacing: 12) {
                    InfoPill(text: "Judge: \(judgeDisplayName)", symbol: "scales")
                    if config.trainMode == .onlineDPO || config.trainMode == .xpo {
                        InfoPill(text: config.trainMode.title, symbol: "arrow.triangle.2.circlepath")
                    }
                }
                if config.judgeKind == .llm && (config.trainMode == .onlineDPO || config.trainMode == .xpo || config.trainMode == .ppo) {
                    TextField("Judge system prompt", text: $config.judgeSystem, axis: .vertical)
                        .lineLimit(2...5)
                }
                if config.trainMode == .xpo {
                    TextField("Alpha schedule", text: $config.alpha)
                }
            }
            .formBlock()
        }
    }

    private var judgeDisplayName: String {
        let trimmed = config.judge.trimmingCharacters(in: .whitespacesAndNewlines)
        if config.judgeKind == .user { return "User / human" }
        return trimmed.isEmpty ? "Default" : trimmed
    }

    private func applyJudgeKind(_ kind: JudgeKind) {
        switch kind {
        case .user:
            config.judge = "human"
        case .llm:
            let trimmed = config.judge.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty || trimmed.localizedCaseInsensitiveCompare("human") == .orderedSame {
                config.judge = "Qwen/Qwen3-0.6B"
            }
        }
    }
}

private struct DatasetMappingSection: View {
    @Binding var config: TrainingConfig

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle("Dataset Columns")
            HStack {
                TextField("Prompt", text: $config.promptFeature)
                TextField("Completion", text: $config.completionFeature)
            }
            HStack {
                TextField("Chosen", text: $config.chosenFeature)
                TextField("Rejected", text: $config.rejectedFeature)
            }
            HStack {
                TextField("Chat", text: $config.chatFeature)
                TextField("Text", text: $config.textFeature)
            }
            HStack {
                TextField("System", text: $config.systemFeature)
                TextField("Answer", text: $config.answerFeature)
                TextField("Type", text: $config.typeFeature)
            }
            TextField("Preference score", text: $config.preferenceScoreFeature)
        }
        .formBlock()
    }
}

private struct TrainSettingsPaneModifier: ViewModifier {
    let cornerRadius: CGFloat

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)

        content
            .scrollContentBackground(.hidden)
            .background(Color(nsColor: .windowBackgroundColor).opacity(0.72), in: shape)
            .clipShape(shape)
            .contentShape(shape)
            .overlay {
                shape.strokeBorder(.quaternary.opacity(0.28), lineWidth: 1)
                    .allowsHitTesting(false)
            }
            .compositingGroup()
    }
}

private extension View {
    func trainSettingsPane(cornerRadius: CGFloat) -> some View {
        modifier(TrainSettingsPaneModifier(cornerRadius: cornerRadius))
    }
}

private struct QATSection: View {
    @Binding var config: TrainingConfig

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle("QAT")
            ToggleRow("Enable QAT projection", isOn: qatEnableBinding)
            if config.qatEnable {
                HStack {
                    NumberField("Bits", value: $config.qatBits)
                    NumberField("Group", value: $config.qatGroupSize)
                    NumberField("Start", value: $config.qatStartStep)
                    NumberField("Interval", value: $config.qatInterval)
                }
            }
        }
        .formBlock()
    }

    private var qatEnableBinding: Binding<Bool> {
        Binding(
            get: { config.qatEnable },
            set: { isEnabled in
                config.qatEnable = isEnabled
                if isEnabled {
                    config.fuseDequantize = false
                }
            }
        )
    }
}
