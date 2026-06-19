import AppKit
import SwiftUI

/// The Runs page. Two stacked regions:
///
/// 1. **Active runs** — a slim list of in-memory `RunRecord`s the
///    store has appended during this session. Shows what the user
///    kicked off and what is still in flight.
/// 2. **Previous runs** — a card grid of every run discovered on disk
///    in the output root. Newest first. Click a card to open a
///    `RunDetailSheet` modal with the loss curve, the auto-discovered
///    metrics cards, the metrics table, and the full training settings
///    panel for that run.
struct RunsView: View {
    @Bindable var store: AppStore
    @State private var selectedRun: PersistedRun?
    @State private var runPendingDelete: PersistedRun?
    @State private var showingClearAllConfirmation = false
    @State private var deleteErrorTitle = "Could Not Delete Run"
    @State private var deleteErrorMessage = ""
    @State private var showingDeleteError = false

    private var activeSessionRuns: [RunRecord] {
        store.runs.filter { $0.endedAt == nil && ["Running", "Paused"].contains($0.status) }
    }

    private var deletablePersistedRuns: [PersistedRun] {
        store.persistedRuns.filter { !isCurrentRun($0) }
    }

    private var protectedActiveRunCount: Int {
        store.persistedRuns.count - deletablePersistedRuns.count
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header
                activeRunsSection
                previousRunsSection
            }
            .padding(24)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .liquidGlass(cornerRadius: 18)
        .padding(16)
        .navigationTitle("Runs")
        .sheet(item: $selectedRun) { run in
            RunDetailSheet(
                run: run,
                canDelete: !isCurrentRun(run),
                continuationCandidate: store.continuationCandidate(for: run),
                onResume: {
                    selectedRun = nil
                    Task { await store.resumeTraining(from: run) }
                },
                onDelete: {
                    selectedRun = nil
                    runPendingDelete = run
                }
            )
        }
        .alert("Delete Run?", isPresented: deleteConfirmationBinding) {
            Button("Delete", role: .destructive) {
                guard let run = runPendingDelete else { return }
                Task { await delete(run) }
            }
            Button("Cancel", role: .cancel) {
                runPendingDelete = nil
            }
        } message: {
            Text("This permanently deletes the run folder and everything inside it.")
        }
        .alert("Clear All Runs?", isPresented: $showingClearAllConfirmation) {
            Button("Clear All Runs", role: .destructive) {
                Task { await clearAllRuns() }
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text(clearAllConfirmationMessage)
        }
        .alert(deleteErrorTitle, isPresented: $showingDeleteError) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(deleteErrorMessage)
        }
    }

    private var deleteConfirmationBinding: Binding<Bool> {
        Binding(
            get: { runPendingDelete != nil },
            set: { if !$0 { runPendingDelete = nil } }
        )
    }

    private func isCurrentRun(_ run: PersistedRun) -> Bool {
        store.allJobRunners.contains { $0.isRunning && run.folderURL.path == $0.lastRunFolder }
    }

    private func delete(_ run: PersistedRun) async {
        do {
            try await store.deletePersistedRun(run)
            if selectedRun?.id == run.id {
                selectedRun = nil
            }
        } catch {
            deleteErrorTitle = "Could Not Delete Run"
            deleteErrorMessage = error.localizedDescription
            showingDeleteError = true
        }
        runPendingDelete = nil
    }

    private func clearAllRuns() async {
        do {
            try await store.deletePersistedRuns(deletablePersistedRuns)
            selectedRun = nil
        } catch {
            deleteErrorTitle = "Could Not Clear Runs"
            deleteErrorMessage = error.localizedDescription
            showingDeleteError = true
            await store.refreshPersistedRuns()
        }
    }

    private var clearAllConfirmationMessage: String {
        let count = deletablePersistedRuns.count
        if protectedActiveRunCount > 0 {
            return "This permanently deletes \(count) saved run folder\(count == 1 ? "" : "s"). Active run folders are kept until they finish."
        }
        return "This permanently deletes \(count) saved run folder\(count == 1 ? "" : "s") and everything inside."
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            HeaderView(
                title: "Runs",
                subtitle: store.anyJobRunning
                    ? "A job is active"
                    : "History is read from \(store.outputRoot)",
                symbol: "chart.xyaxis.line"
            )
            Spacer()
            Button(role: .destructive) {
                showingClearAllConfirmation = true
            } label: {
                Label("Clear All Runs", systemImage: "trash")
            }
            .disabled(deletablePersistedRuns.isEmpty || store.isRefreshingPersistedRuns || store.isDeletingPersistedRuns)
            .buttonStyle(.bordered)
            .help(deletablePersistedRuns.isEmpty ? "No saved run folders can be deleted" : "Delete all saved run folders")
            Button {
                Task { await store.refreshPersistedRuns() }
            } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .disabled(store.isRefreshingPersistedRuns || store.isDeletingPersistedRuns)
            .buttonStyle(.bordered)
        }
    }

    // MARK: - Active runs

    @ViewBuilder
    private var activeRunsSection: some View {
        if !activeSessionRuns.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                SectionTitle("Active Runs")
                Text("Jobs currently running in this session.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                VStack(spacing: 0) {
                    ForEach(activeSessionRuns) { run in
                        activeRunRow(run)
                        if run.id != activeSessionRuns.last?.id {
                            Divider().background(Color.white.opacity(0.05))
                        }
                    }
                }
                .padding(8)
                .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
            }
            .formBlock()
        }
    }

    private func activeRunRow(_ run: RunRecord) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(run.title).font(.headline)
                Spacer()
                Text(run.status)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(run.status == "Running" ? .green : .secondary)
            }
            Text(run.command)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(2)
            Text(run.startedAt, style: .date)
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 6)
    }

    // MARK: - Previous runs

    @ViewBuilder
    private var previousRunsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                SectionTitle("Previous Runs")
                Spacer()
                if !store.persistedRuns.isEmpty {
                    Text("\(store.persistedRuns.count) on disk")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Text("Discovered from the output folder. Click a card to inspect its metrics and settings.")
                .font(.callout)
                .foregroundStyle(.secondary)

            if store.persistedRuns.isEmpty {
                ContentUnavailableView(
                    "No Runs on Disk Yet",
                    systemImage: "tray",
                    description: Text("Start a training job — its run folder will appear here after the first report lands.")
                )
                .frame(maxWidth: .infinity, minHeight: 220)
            } else {
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 280, maximum: 360), spacing: 14)],
                    spacing: 14
                ) {
                    ForEach(store.persistedRuns) { run in
                        RunCard(
                            run: run,
                            canDelete: !isCurrentRun(run),
                            continuationCandidate: store.continuationCandidate(for: run),
                            onOpen: {
                                selectedRun = run
                            },
                            onResume: {
                                Task { await store.resumeTraining(from: run) }
                            },
                            onDelete: {
                                runPendingDelete = run
                            }
                        )
                    }
                }
            }
        }
        .formBlock()
    }
}

// MARK: - Card

/// Compact summary card for a single previous run. Shows the algorithm
/// + model, a sparkline of the loss curve, the final loss value, and
/// the start date. Click anywhere on the card to open the detail sheet.
private struct RunCard: View {
    let run: PersistedRun
    let canDelete: Bool
    let continuationCandidate: TrainingResumeCandidate?
    let onOpen: () -> Void
    let onResume: () -> Void
    let onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            sparkline
            footer
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.18), in: RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color.white.opacity(0.06), lineWidth: 1)
        )
        .contentShape(RoundedRectangle(cornerRadius: 10))
        .onTapGesture(perform: onOpen)
        .contextMenu {
            Button {
                onOpen()
            } label: {
                Label("Open Details", systemImage: "arrow.up.right.square")
            }
            Button {
                onResume()
            } label: {
                Label("Continue Training", systemImage: "play.circle")
            }
            .disabled(continuationCandidate == nil)
            Button(role: .destructive) {
                onDelete()
            } label: {
                Label("Delete Run", systemImage: "trash")
            }
            .disabled(!canDelete)
        }
        .help("Open run details")
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                kindBadge
                Spacer()
                if let continuationCandidate {
                    Button {
                        onResume()
                    } label: {
                        Image(systemName: "play.circle")
                            .font(.caption)
                    }
                    .buttonStyle(.plain)
                    .help(continueHelp(for: continuationCandidate))
                }
                Button(role: .destructive) {
                    onDelete()
                } label: {
                    Image(systemName: "trash")
                        .font(.caption)
                }
                .buttonStyle(.plain)
                .disabled(!canDelete)
                .help(canDelete ? "Delete run folder" : "Cannot delete the active run")
                Image(systemName: "arrow.up.right.square")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Text(run.title)
                .font(.subheadline.weight(.semibold))
                .lineLimit(1)
            if let spec = run.spec {
                Text(spec.model)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
    }

    private var kindBadge: some View {
        let label: String
        let color: Color
        switch run.kind {
        case .training:
            label = run.spec?.trainMode.title ?? "Training"
            color = .orange
        case .synthetic:
            label = "Synthetic"
            color = .purple
        case .hfUpload:
            label = "HF Upload"
            color = .blue
        }
        return Text(label)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.18), in: Capsule())
            .foregroundStyle(color)
    }

    @ViewBuilder
    private var sparkline: some View {
        let train = run.metrics.compactMap(\.loss)
        let val = run.metrics.compactMap(\.validationLoss)
        if train.isEmpty {
            ZStack {
                RoundedRectangle(cornerRadius: 6)
                    .fill(Color.white.opacity(0.04))
                Text("No metrics captured")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            .frame(height: 56)
        } else {
            MiniSparkline(trainValues: train, valValues: val)
                .frame(height: 56)
        }
    }

    private var footer: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            VStack(alignment: .leading, spacing: 1) {
                Text("Final loss")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                Text(run.finalLoss.map(formatLoss) ?? "—")
                    .font(.system(.callout, design: .monospaced))
            }
            if let val = run.finalValidationLoss {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Val")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                    Text(formatLoss(val))
                        .font(.system(.callout, design: .monospaced))
                        .foregroundStyle(.blue)
                }
            }
            Spacer()
            Text(run.createdAt, format: .dateTime.month().day().hour().minute())
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    private func formatLoss(_ v: Double) -> String {
        String(format: "%.3f", v)
    }

    private func continueHelp(for candidate: TrainingResumeCandidate) -> String {
        if let step = candidate.step {
            return "Continue training from checkpoint step \(step)"
        }
        return "Continue training from saved adapter weights"
    }
}

// MARK: - Detail sheet

/// Modal that renders everything we know about a previous run: the
/// loss curve (with validation overlay), the same auto-discovered
/// metric cards the Live Metrics page shows, the raw metrics table,
/// the recent trainer reports, and a settings panel summarising the
/// exact `TrainingConfig` that was used. Re-uses `ChartCard`,
/// `MultiLineChart`, `MetricCardGroup`, and `MetricTableRow` from the
/// Live Metrics view so the visual language stays consistent.
private struct RunDetailSheet: View {
    let run: PersistedRun
    let canDelete: Bool
    let continuationCandidate: TrainingResumeCandidate?
    let onResume: () -> Void
    let onDelete: () -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    lossCard
                    autoDiscoveredGrid
                    metricsTableCard
                    settingsCard
                    syntheticSamplesCard
                    recentReportsCard
                }
                .padding(20)
            }
        }
        .frame(minWidth: 820, idealWidth: 980, minHeight: 640, idealHeight: 760)
    }

    // MARK: Header

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(run.title)
                    .font(.title2.weight(.semibold))
                Text(run.folderURL.path)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer()
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([run.folderURL])
            } label: {
                Label("Open in Finder", systemImage: "folder")
            }
            .buttonStyle(.bordered)
            if let continuationCandidate {
                Button {
                    onResume()
                    dismiss()
                } label: {
                    Label("Continue", systemImage: "play.circle")
                }
                .buttonStyle(.borderedProminent)
                .help(continueHelp(for: continuationCandidate))
            }
            Button(role: .destructive) {
                onDelete()
            } label: {
                Label("Delete", systemImage: "trash")
            }
            .buttonStyle(.bordered)
            .disabled(!canDelete)
            .help(canDelete ? "Delete this run folder" : "Cannot delete the active run")
            Button {
                dismiss()
            } label: {
                Text("Done")
            }
            .buttonStyle(.bordered)
            .keyboardShortcut(.defaultAction)
        }
        .padding(20)
    }

    private func continueHelp(for candidate: TrainingResumeCandidate) -> String {
        if let step = candidate.step {
            return "Continue from checkpoint step \(step)"
        }
        return "Continue from saved adapter weights"
    }

    // MARK: Loss chart

    @ViewBuilder
    private var lossCard: some View {
        let series = buildLossSeries()
        if !series.isEmpty {
            let hasVal = series.contains { $0.id == "val_loss" }
            ChartCard(
                title: "Loss",
                subtitle: hasVal
                    ? "Training (solid) and validation (dashed) loss across iterations."
                    : "Training loss across iterations."
            ) {
                MultiLineChart(series: series, yLabel: "loss")
            }
            .frame(minHeight: 200)
        }
    }

    /// Pulls (loss, step) pairs for train + val from the persisted
    /// metrics. Returns an empty array when the run never produced a
    /// loss sample so the caller can decide not to render a card at
    /// all.
    private func buildLossSeries() -> [MultiLineChart.Series] {
        let trainPairs: [(Double, Int)] = run.metrics.compactMap { m in
            m.loss.map { ($0, m.step) }
        }
        let valPairs: [(Double, Int)] = run.metrics.compactMap { m in
            m.validationLoss.map { ($0, m.step) }
        }
        var series: [MultiLineChart.Series] = []
        if !trainPairs.isEmpty {
            series.append(
                MultiLineChart.Series(
                    id: "loss",
                    label: "Train",
                    color: .orange,
                    values: trainPairs.map(\.0),
                    xSteps: trainPairs.map(\.1)
                )
            )
        }
        if !valPairs.isEmpty {
            series.append(
                MultiLineChart.Series(
                    id: "val_loss",
                    label: "Validation",
                    color: .blue,
                    values: valPairs.map(\.0),
                    isDashed: true,
                    xSteps: valPairs.map(\.1)
                )
            )
        }
        return series
    }

    // MARK: Auto-discovered grid

    @ViewBuilder
    private var autoDiscoveredGrid: some View {
        let groups = MetricCardGroup.compute(from: run.metrics)
        if !groups.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                SectionTitle("Auto-Discovered Metrics")
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 320, maximum: 540), spacing: 16)],
                    spacing: 16
                ) {
                    ForEach(groups) { group in
                        ChartCard(title: group.title, subtitle: group.subtitle) {
                            MultiLineChart(
                                series: group.series,
                                yLabel: nil,
                                emptyMessage: "Collecting samples…"
                            )
                        }
                        .frame(minHeight: 160)
                    }
                }
            }
            .formBlock()
        }
    }

    // MARK: Metrics table

    @ViewBuilder
    private var metricsTableCard: some View {
        let rows = MetricTableRow.compute(from: run.metrics)
        if !rows.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                SectionTitle("All Metrics")
                VStack(spacing: 0) {
                    tableHeader
                    Divider().background(Color.white.opacity(0.08))
                    ForEach(rows) { row in
                        tableRow(row)
                        if row.id != rows.last?.id {
                            Divider().background(Color.white.opacity(0.04))
                        }
                    }
                }
                .padding(8)
                .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
            }
            .formBlock()
        }
    }

    private var tableHeader: some View {
        HStack {
            Text("Key").frame(maxWidth: .infinity, alignment: .leading)
            Text("Latest").frame(width: 90, alignment: .trailing)
            Text("Min").frame(width: 80, alignment: .trailing)
            Text("Max").frame(width: 80, alignment: .trailing)
            Text("N").frame(width: 40, alignment: .trailing)
        }
        .font(.caption.weight(.semibold))
        .foregroundStyle(.secondary)
        .padding(.vertical, 4)
    }

    private func tableRow(_ row: MetricTableRow) -> some View {
        HStack {
            HStack(spacing: 4) {
                Image(systemName: row.isValidation ? "checkmark.seal.fill" : "bolt.fill")
                    .font(.caption2)
                    .foregroundStyle(row.isValidation ? .blue.opacity(0.8) : .orange.opacity(0.8))
                Text(row.displayName)
                    .font(.system(.caption, design: .monospaced))
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Text(formatNumber(row.latest))
                .font(.caption.monospacedDigit())
                .frame(width: 90, alignment: .trailing)
            Text(formatNumber(row.min))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .trailing)
            Text(formatNumber(row.max))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .trailing)
            Text("\(row.count)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 40, alignment: .trailing)
        }
        .padding(.vertical, 4)
    }

    private func formatNumber(_ v: Double) -> String {
        if abs(v) >= 1000 || (abs(v) < 0.01 && v != 0) {
            return String(format: "%.2e", v)
        }
        return String(format: "%.3f", v)
    }

    // MARK: Settings

    @ViewBuilder
    private var settingsCard: some View {
        if let syntheticSpec = run.syntheticSpec {
            VStack(alignment: .leading, spacing: 8) {
                SectionTitle("Dataset Run Settings")
                SyntheticSettingsGrid(spec: syntheticSpec)
            }
            .formBlock()
        } else if run.kind == .synthetic {
            VStack(alignment: .leading, spacing: 8) {
                SectionTitle("Dataset Run Settings")
                Text("The synthetic_spec.json for this folder could not be decoded. The raw file is still on disk if you want to inspect it.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            .formBlock()
        } else if let spec = run.spec {
            VStack(alignment: .leading, spacing: 8) {
                SectionTitle("Training Settings")
                SettingsGrid(spec: spec)
            }
            .formBlock()
        } else if run.kind == .training {
            VStack(alignment: .leading, spacing: 8) {
                SectionTitle("Training Settings")
                Text("The run_spec.json for this folder could not be decoded. The raw file is still on disk if you want to inspect it.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            .formBlock()
        }
    }

    @ViewBuilder
    private var syntheticSamplesCard: some View {
        if run.kind == .synthetic {
            VStack(alignment: .leading, spacing: 8) {
                SectionTitle("Dataset Samples")
                if run.syntheticSamples.isEmpty {
                    Text("No generated JSONL samples were found in this run folder yet.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                } else {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(run.syntheticSamples) { sample in
                            syntheticSampleBlock(sample)
                        }
                    }
                    .padding(10)
                    .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
                }
            }
            .formBlock()
        }
    }

    private func syntheticSampleBlock(_ sample: SyntheticDatasetSample) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Sample \(sample.index)")
                    .font(.caption.weight(.semibold))
                Spacer()
                Text(sample.sourceFile)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
            ForEach(sample.fields) { field in
                VStack(alignment: .leading, spacing: 3) {
                    Text(field.key)
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .textCase(.uppercase)
                    Text(shortened(field.value, limit: 1_800))
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .lineLimit(12)
                }
            }
        }
        .padding(10)
        .background(.quaternary.opacity(0.18), in: RoundedRectangle(cornerRadius: 8))
    }

    private func shortened(_ value: String, limit: Int) -> String {
        guard value.count > limit else { return value }
        let index = value.index(value.startIndex, offsetBy: limit)
        return String(value[..<index]) + "\n…"
    }

    // MARK: Recent reports

    @ViewBuilder
    private var recentReportsCard: some View {
        if !run.metrics.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                SectionTitle("Recent Reports")
                let recents = Array(run.metrics.suffix(5).reversed())
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(recents) { metric in
                        rawLineBlock(metric)
                    }
                }
                .padding(10)
                .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
            }
            .formBlock()
        }
    }

    private func rawLineBlock(_ metric: TrainingMetric) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Iter \(metric.step)")
                    .font(.caption.weight(.semibold))
                Spacer()
                if let loss = metric.loss {
                    Text("loss \(String(format: "%.3f", loss))")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if let mem = metric.memoryGB {
                    Text(String(format: "%.2f GB", mem))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            Text(metric.rawLine)
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
                .lineLimit(8)
                .foregroundStyle(.secondary)
        }
    }
}

// MARK: - Settings grid

private struct SyntheticSettingsGrid: View {
    let spec: SyntheticConfig

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            settingsGroup("Dataset") {
                row("Type", spec.kind.title)
                row("Source dataset", spec.datasetPath)
                row("Output folder", spec.outputDir)
                if !spec.resumeOutputDir.isEmpty {
                    row("Resume output", spec.resumeOutputDir)
                }
                row("Samples", "\(spec.numSamples)")
                row("Validation split", spec.validSplit)
                row("Test split", spec.testSplit)
            }
            settingsGroup("Provider") {
                row("Backend", spec.backend.title)
                row("Base URL", spec.baseURL)
                if spec.kind == .sft {
                    row("Model", spec.model)
                } else {
                    row("Base model", spec.baseModel)
                    row("Teacher model", spec.teacherModel)
                    row("Generation target", spec.dpoGenerationTarget.title)
                }
            }
            settingsGroup("Generation") {
                row("Batch size", "\(spec.batchSize)")
                row("Max concurrent", "\(spec.maxConcurrent)")
                row("Seed", "\(spec.seed)")
                row("System prompt", spec.systemPrompt)
                row("Include system prompt", spec.includeSystemPrompt ? "on" : "off")
                row("Use ground truth", spec.useGroundTruth ? "on" : "off")
            }
            if spec.kind == .sft {
                settingsGroup("Conversation") {
                    row("Multiturn", spec.multiturn ? "on" : "off")
                    row("Max turns", "\(spec.maxTurns)")
                    row("Multiturn percentile", String(format: "%.2f", spec.multiturnPercentile))
                    row("Human role model", spec.humanRoleModel)
                }
            }
            if spec.useGenerationSettings {
                settingsGroup("Sampling") {
                    row("Max tokens", "\(spec.maxTokens)")
                    row("Temperature", String(format: "%.2f", spec.temperature))
                    row("Top-p", String(format: "%.2f", spec.topP))
                    row("Min-p", String(format: "%.2f", spec.minP))
                    row("Top-k", "\(spec.topK)")
                    row("Min tokens to keep", "\(spec.minTokensToKeep)")
                    row("XTC probability", String(format: "%.2f", spec.xtcProbability))
                    row("XTC threshold", String(format: "%.2f", spec.xtcThreshold))
                }
            }
        }
    }

    @ViewBuilder
    private func settingsGroup<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            VStack(alignment: .leading, spacing: 4) {
                content()
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary.opacity(0.18), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private func row(_ key: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(key)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 180, alignment: .leading)
            Text(value.isEmpty ? "-" : value)
                .font(.caption.monospacedDigit())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// Two-column key/value listing of the run's `TrainingConfig`. Groups
/// the fields into headed sections (Model & Data, Optimisation,
/// Algorithm-specific, …) so a 60-line config doesn't become an
/// unreadable wall of rows.
private struct SettingsGrid: View {
    let spec: TrainingConfig

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            settingsGroup("Model & Data") {
                row("Model", spec.model)
                row("Dataset", spec.data)
                row("Adapter path", spec.adapterPath)
            }
            settingsGroup("Algorithm") {
                row("Mode", spec.trainMode.title)
                row("Family", spec.trainMode.family)
                row("Train type", spec.trainType.title)
                row("Optimizer", spec.optimizer.title)
                row("Quantization", spec.quantization.title)
                if spec.trainMode.needsReference, !spec.referenceModelPath.isEmpty {
                    row("Reference model", spec.referenceModelPath)
                }
                if spec.trainMode.needsJudge, !spec.judge.isEmpty {
                    row("Judge", spec.judge)
                }
            }
            settingsGroup("Optimisation") {
                row("Learning rate", String(format: "%.3e", spec.learningRate))
                row("LR schedule", scheduleLabel)
                row("Batch size", "\(spec.batchSize)")
                row("Grad accumulation", "\(spec.gradientAccumulationSteps)")
                row(itersOrEpochsKey, itersOrEpochsValue)
                row("Val batches", "\(spec.valBatches)")
                row("Max seq length", "\(spec.maxSeqLength)")
            }
            settingsGroup("LoRA Parameters") {
                row("Rank", "\(spec.rank)")
                row("Scale", String(format: "%.1f", spec.scale))
                row("Dropout", String(format: "%.3f", spec.dropout))
            }
            settingsGroup("Reporting & Saving") {
                row("Steps per report", "\(spec.stepsPerReport)")
                row("Steps per eval", "\(spec.stepsPerEval)")
                row("Save every", "\(spec.saveEvery)")
            }
            settingsGroup("Toggles") {
                row("Grad checkpoint", spec.gradCheckpoint ? "on" : "off")
                row("Efficient long context", spec.efficientLongContext ? "on" : "off")
                row("Mask prompt", spec.maskPrompt ? "on" : "off")
                row("Fuse", spec.fuse ? "on" : "off")
            }
            algorithmSpecificGroup
            qatGroup
        }
    }

    // MARK: - Group builders

    @ViewBuilder
    private func settingsGroup<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            VStack(alignment: .leading, spacing: 4) {
                content()
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary.opacity(0.18), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    @ViewBuilder
    private var algorithmSpecificGroup: some View {
        switch spec.trainMode {
        case .dpo, .cpo, .orpo, .onlineDPO, .xpo, .rlhfReinforce, .ppo:
            settingsGroup("Preference / RL") {
                row("β (beta)", String(format: "%.3f", spec.beta))
                if spec.trainMode == .dpo || spec.trainMode == .cpo {
                    row("Loss type", spec.dpoCpoLossType)
                }
                if spec.trainMode == .cpo || spec.trainMode == .orpo {
                    row("δ (delta)", String(format: "%.1f", spec.delta))
                }
                if spec.trainMode.needsJudge, !spec.alpha.isEmpty {
                    row("α (alpha)", spec.alpha)
                }
                if spec.trainMode == .rlhfReinforce {
                    row("Reward scaling", String(format: "%.2f", spec.rewardScaling))
                }
            }
        case .grpo:
            settingsGroup("GRPO") {
                row("Group size", "\(spec.groupSize)")
                row("ε (epsilon)", String(format: "%.4f", spec.epsilon))
                if !spec.epsilonHigh.isEmpty {
                    row("ε high", spec.epsilonHigh)
                }
                row("Loss type", spec.grpoLossType)
                row("Max completion length", "\(spec.maxCompletionLength)")
                row("Temperature", String(format: "%.2f", spec.temperature))
                row("Top-p", String(format: "%.2f", spec.topP))
                row("Top-k", "\(spec.topK)")
                row("Min-p", String(format: "%.2f", spec.minP))
            }
        case .sft:
            EmptyView()
        }
    }

    @ViewBuilder
    private var qatGroup: some View {
        if spec.qatEnable {
            settingsGroup("Quantization-Aware Training") {
                row("Bits", "\(spec.qatBits)")
                row("Group size", "\(spec.qatGroupSize)")
                row("Start step", "\(spec.qatStartStep)")
                row("Interval", "\(spec.qatInterval)")
            }
        }
    }

    // MARK: - Row + helpers

    private func row(_ key: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(key)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 180, alignment: .leading)
            Text(value.isEmpty ? "—" : value)
                .font(.caption.monospacedDigit())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var scheduleLabel: String {
        switch spec.learningRateSchedule {
        case .constant:
            return "constant"
        case .cosineDecay:
            return "cosine decay (warmup \(spec.lrWarmupSteps), init \(String(format: "%.2e", spec.lrWarmupInit)), final \(String(format: "%.2e", spec.lrFinal)))"
        }
    }

    private var itersOrEpochsKey: String {
        spec.epochs > 0 ? "Epochs" : "Iterations"
    }

    private var itersOrEpochsValue: String {
        spec.epochs > 0 ? "\(spec.epochs)" : "\(spec.iters)"
    }
}
