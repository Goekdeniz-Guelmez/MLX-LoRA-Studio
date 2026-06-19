import SwiftUI
#if os(macOS)
import AppKit
#endif

struct HeaderView: View {
    let title: String
    let subtitle: String
    let symbol: String

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: symbol)
                .font(.title2)
                .frame(width: 42, height: 42)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.largeTitle.bold())
                Text(subtitle)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
    }
}

struct SectionTitle: View {
    let title: String

    init(_ title: String) {
        self.title = title
    }

    var body: some View {
        Text(title)
            .font(.headline)
    }
}

struct InfoPill: View {
    let text: String
    let symbol: String
    var openPath: String?

    var body: some View {
        if let openPath {
            Button {
                openFolder(openPath)
            } label: {
                pillLabel
            }
            .buttonStyle(.plain)
            .help("Open in Finder")
        } else {
            pillLabel
        }
    }

    private var pillLabel: some View {
        Label(text, systemImage: symbol)
            .font(.callout)
            .foregroundStyle(.secondary)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .liquidGlass(cornerRadius: 999, interactive: true)
    }

    private func openFolder(_ path: String) {
        #if os(macOS)
        let expanded = NSString(string: path).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded, isDirectory: true)
        let target = FileManager.default.fileExists(atPath: url.path)
            ? url
            : url.deletingLastPathComponent()
        NSWorkspace.shared.open(target)
        #endif
    }
}

struct NumberField: View {
    let title: String
    @Binding var value: Int

    init(_ title: String, value: Binding<Int>) {
        self.title = title
        self._value = value
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
            TextField(title, value: $value, format: .number)
                .textFieldStyle(.roundedBorder)
                .font(.system(.body, design: .monospaced))
        }
        .frame(minWidth: 88, maxWidth: .infinity, alignment: .leading)
    }
}

struct FloatingField: View {
    let title: String
    @Binding var value: Double

    init(_ title: String, value: Binding<Double>) {
        self.title = title
        self._value = value
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
            TextField(title, value: $value, format: .number)
                .textFieldStyle(.roundedBorder)
                .font(.system(.body, design: .monospaced))
        }
        .frame(minWidth: 88, maxWidth: .infinity, alignment: .leading)
    }
}

struct ToggleRow: View {
    let title: String
    @Binding var isOn: Bool

    init(_ title: String, isOn: Binding<Bool>) {
        self.title = title
        self._isOn = isOn
    }

    var body: some View {
        Toggle(title, isOn: $isOn)
            .toggleStyle(.switch)
    }
}

struct SystemPromptPicker: View {
    @Binding var text: String
    let placeholder: String

    @Environment(AppStore.self) private var store
    @State private var showingAddSheet = false
    @State private var newPrompt = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top, spacing: 6) {
                Image(systemName: "text.quote")
                    .foregroundStyle(.secondary)
                    .padding(.top, 6)
                ZStack(alignment: .topLeading) {
                    TextEditor(text: $text)
                        .font(.system(.body, design: .monospaced))
                        .scrollContentBackground(.hidden)
                        .padding(8)
                        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .strokeBorder(.quaternary.opacity(0.6), lineWidth: 1)
                        )

                    if text.isEmpty {
                        Text(placeholder)
                            .font(.system(.body, design: .monospaced))
                            .foregroundStyle(.tertiary)
                            .padding(.horizontal, 13)
                            .padding(.vertical, 10)
                            .allowsHitTesting(false)
                    }
                }
                .frame(minHeight: 220, maxHeight: 340)
                Menu {
                    menuContent
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
            }

            Text(text.isEmpty
                 ? "Write a system prompt here, or load one from the dropdown."
                 : "\(text.count) character\(text.count == 1 ? "" : "s") · Save or load reusable prompts from the dropdown.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .sheet(isPresented: $showingAddSheet) {
            AddSystemPromptSheet(
                initial: newPrompt,
                onCommit: { prompt in
                    let clean = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !clean.isEmpty else { return }
                    store.addCustomSystemPrompt(clean)
                    text = clean
                    newPrompt = ""
                },
                onCancel: { newPrompt = "" }
            )
        }
    }

    @ViewBuilder
    private var menuContent: some View {
        Section("Saved prompts (\(store.customSystemPrompts.count))") {
            if store.customSystemPrompts.isEmpty {
                Text("No saved prompts yet").foregroundStyle(.secondary)
            } else {
                ForEach(store.customSystemPrompts, id: \.self) { prompt in
                    Button {
                        text = prompt
                    } label: {
                        Label(promptPreview(prompt), systemImage: "text.quote")
                    }
                }
            }
        }

        Section {
            Button {
                store.addCustomSystemPrompt(text)
            } label: {
                Label("Save current prompt", systemImage: "bookmark")
            }
            .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

            Button {
                newPrompt = ""
                showingAddSheet = true
            } label: {
                Label("Add custom prompt...", systemImage: "plus.circle")
            }
        }

        if !store.customSystemPrompts.isEmpty {
            Section("Remove saved prompt") {
                ForEach(store.customSystemPrompts, id: \.self) { prompt in
                    Button(role: .destructive) {
                        store.removeCustomSystemPrompt(prompt)
                    } label: {
                        Label(promptPreview(prompt), systemImage: "trash")
                    }
                }
            }
        }
    }

    private func promptPreview(_ prompt: String) -> String {
        let singleLine = prompt
            .split(whereSeparator: \.isNewline)
            .joined(separator: " ")
        if singleLine.count <= 64 { return singleLine }
        return String(singleLine.prefix(61)) + "..."
    }
}

private struct AddSystemPromptSheet: View {
    @State var initial: String
    let onCommit: (String) -> Void
    let onCancel: () -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Add system prompt")
                .font(.headline)
            Text("Paste a reusable system prompt. It will be saved locally and loaded into the system prompt field when selected.")
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            ZStack(alignment: .topLeading) {
                TextEditor(text: $initial)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(.primary)
                    .scrollContentBackground(.hidden)
                    .padding(8)
                    .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .strokeBorder(.separator.opacity(0.8), lineWidth: 1)
                    )

                if initial.isEmpty {
                    Text("You are a helpful assistant...")
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 16)
                        .allowsHitTesting(false)
                }
            }
            .frame(width: 540, height: 260)
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) {
                    onCancel()
                    dismiss()
                }
                Button("Add") {
                    onCommit(initial)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(initial.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 580)
    }
}

extension View {
    func formBlock() -> some View {
        self
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .liquidGlass(cornerRadius: 12)
    }
}

/// Compact glass card showing a single labelled value. Used in both the
/// Live Run panel and the Live Metrics page for the headline summary
/// row at the top of each.
struct StatTile: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.headline.monospacedDigit())
                .lineLimit(1)
                .minimumScaleFactor(0.75)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .liquidGlass(cornerRadius: 10)
    }
}

struct RunProgressBar: View {
    let runner: PythonJobRunner

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { context in
            if let total = runner.progressTotal, total > 0 {
                let current = min(max(runner.progressCurrent ?? 0, 0), total)
                let fraction = min(max(Double(current) / Double(total), 0), 1)

                VStack(alignment: .leading, spacing: 8) {
                    HStack(alignment: .firstTextBaseline) {
                        Text(runner.progressLabel.isEmpty ? "Progress" : runner.progressLabel)
                            .font(.caption.weight(.semibold))
                        Text("\(current) / \(total)")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(etaText(current: current, total: total, now: context.date))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }

                    ProgressView(value: Double(current), total: Double(total))
                        .progressViewStyle(.linear)
                        .tint(progressTint(fraction: fraction))

                    HStack {
                        Text(String(format: "%.0f%% complete", fraction * 100))
                        Spacer()
                        Text(rateText(current: current, now: context.date))
                    }
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
                .animation(.easeInOut(duration: 0.2), value: current)
                .animation(.easeInOut(duration: 0.2), value: runner.isRunning)
            }
        }
    }

    private func etaText(current: Int, total: Int, now: Date) -> String {
        guard runner.isRunning else {
            return current >= total ? "Done" : "Stopped"
        }
        if runner.isPaused {
            return "Paused"
        }
        guard current > 0,
              let startedAt = runner.startedAt else {
            return "ETA estimating"
        }
        let elapsed = max(now.timeIntervalSince(startedAt), 0)
        guard elapsed > 0 else { return "ETA estimating" }
        let secondsPerUnit = elapsed / Double(current)
        let remaining = max(Double(total - current) * secondsPerUnit, 0)
        return "ETA \(formatDuration(remaining))"
    }

    private func rateText(current: Int, now: Date) -> String {
        guard current > 0,
              let startedAt = runner.startedAt else {
            return "— /s"
        }
        let elapsed = max(now.timeIntervalSince(startedAt), 0.001)
        return String(format: "%.2f /s", Double(current) / elapsed)
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        let totalSeconds = max(Int(seconds.rounded()), 0)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let seconds = totalSeconds % 60
        if hours > 0 {
            return String(format: "%dh %02dm", hours, minutes)
        }
        if minutes > 0 {
            return String(format: "%dm %02ds", minutes, seconds)
        }
        return String(format: "%ds", seconds)
    }

    private func progressTint(fraction: Double) -> Color {
        if fraction >= 1 { return .green }
        return .accentColor
    }
}

// MARK: - Live Memory Card

struct LiveMemoryCard: View {
    let snapshot: LiveMemoryMonitor.Snapshot
    let estimate: MemoryEstimator.Report
    @State private var expanded: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            header
            if expanded {
                Divider().opacity(0.35)
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 6) {
                        Image(systemName: "memorychip")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Text("\(snapshot.usedMemoryString) used of \(snapshot.totalMemoryString)")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    usageBar
                    Text("Training estimate")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .padding(.top, 2)
                    Text(estimate.summary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    HStack(spacing: 6) {
                        Image(systemName: estimate.verdict.symbol)
                            .font(.caption2)
                            .foregroundStyle(estimateColor)
                        Text("\(estimate.estimatedPeakString) estimated peak")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    if let suggestion = estimate.suggestion {
                        HStack(alignment: .top, spacing: 6) {
                            Image(systemName: "lightbulb")
                                .font(.caption2)
                                .foregroundStyle(.tint)
                            Text(suggestion)
                                .font(.caption)
                                .foregroundStyle(.primary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .liquidGlass(cornerRadius: 8)
                    }
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .liquidGlass(cornerRadius: 12)
        .contentShape(Rectangle())
        .onTapGesture {
            withAnimation(.easeInOut(duration: 0.18)) { expanded.toggle() }
        }
        .help(expanded ? "Click to collapse" : "Click for live memory details")
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: liveSymbol)
                .foregroundStyle(liveColor)
                .font(.callout)
            VStack(alignment: .leading, spacing: 1) {
                Text(snapshot.chipLabel)
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                Text("Live memory")
                    .font(.caption2)
                    .foregroundStyle(liveColor)
            }
            Spacer(minLength: 4)
            Text(snapshot.usedMemoryString)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Image(systemName: expanded ? "chevron.up" : "chevron.down")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    private var usageBar: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(.quaternary.opacity(0.4))
                Capsule()
                    .fill(liveColor.opacity(0.85))
                    .frame(width: max(2, proxy.size.width * CGFloat(min(snapshot.usedRatio, 1.0))))
            }
        }
        .frame(height: 4)
    }

    private var liveSymbol: String {
        if snapshot.usedRatio >= 0.90 { return "xmark.octagon.fill" }
        if snapshot.usedRatio >= 0.75 { return "exclamationmark.triangle.fill" }
        return "checkmark.circle.fill"
    }

    private var liveColor: Color {
        if snapshot.usedRatio >= 0.90 { return .red }
        if snapshot.usedRatio >= 0.75 { return .orange }
        return .green
    }

    private var estimateColor: Color {
        switch estimate.verdict {
        case .likelyFits: .green
        case .risky: .orange
        case .tooLarge: .red
        }
    }
}

// MARK: - Liquid Glass

struct LiquidGlassModifier: ViewModifier {
    var cornerRadius: CGFloat = 16
    var interactive = false

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)

        if #available(macOS 26.0, *) {
            // Keep the Liquid Glass surface clear enough for the animated
            // background to refract through the panels, then add a light rim
            // so the surface still reads as glass instead of flat blur.
            if interactive {
                content
                    .glassEffect(.clear.interactive(), in: shape)
                    .overlay(liquidGlassRim(shape))
            } else {
                content
                    .glassEffect(.clear, in: shape)
                    .overlay(liquidGlassRim(shape))
            }
        } else {
            // Older OSes can't do real glass — fall back to an ultra-thin
            // material so the same see-through feel is approximated.
            content
                .background(.ultraThinMaterial, in: shape)
                .overlay(
                    shape.strokeBorder(.quaternary.opacity(0.25), lineWidth: 1)
                )
        }
    }

    private func liquidGlassRim(_ shape: RoundedRectangle) -> some View {
        shape
            .strokeBorder(
                LinearGradient(
                    colors: [
                        .white.opacity(0.42),
                        .white.opacity(0.12),
                        .white.opacity(0.28)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                lineWidth: 0.8
            )
            .blendMode(.screen)
            .allowsHitTesting(false)
    }
}

extension View {
    func liquidGlass(cornerRadius: CGFloat = 16, interactive: Bool = false) -> some View {
        modifier(LiquidGlassModifier(cornerRadius: cornerRadius, interactive: interactive))
    }
}

// MARK: - MultiLineChart
// A small overlay-friendly line chart. Each series is normalised to its
// own min/max so two metrics on wildly different scales (rewards at 0..1
// and KL at 0..0.001) don't squash each other. Validation series render
// as dashed lines via `isDashed`. The legend on the right shows the
// series label and the most recent value, color-matched.
//
// Hovering the plot area snaps to the nearest sample index and shows a
// vertical guide + a colour-matched dot on each series + a floating
// callout listing every series' value at that step. The x label is
// either an absolute step number (when `xSteps` is provided) or the raw
// sample index.

struct MultiLineChart: View {
    struct Series: Identifiable, Equatable {
        let id: String
        let label: String
        let color: Color
        let values: [Double]
        var isDashed: Bool = false
        /// Optional per-sample step numbers. When non-nil, the hover
        /// callout shows `step N`; otherwise it falls back to the raw
        /// sample index. Per-series (not per-chart) so a series whose
        /// values are sparse — e.g. a validation series that lags
        /// training — can carry its own x labels.
        var xSteps: [Int]? = nil

        var latest: Double? { values.last }
    }

    let series: [Series]
    var yLabel: String? = nil
    var emptyMessage: String = "Awaiting data."

    @State private var hoverStep: Int? = nil

    var body: some View {
        GeometryReader { proxy in
            let allValues = series.flatMap(\.values)
            if series.isEmpty || series.allSatisfy(\.values.isEmpty) {
                placeholder
            } else if allValues.count < 2 {
                placeholder
            } else {
                chart(in: proxy.size)
            }
        }
    }

    private var placeholder: some View {
        VStack {
            Spacer(minLength: 0)
            Text(emptyMessage)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 12)
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func chart(in size: CGSize) -> some View {
        let plotWidth = max(size.width - legendWidth, 1)
        let plotHeight = max(size.height, 1)
        let plotRect = CGRect(x: 0, y: 0, width: plotWidth, height: plotHeight)
        // Longest series drives the x-axis sample count so the hover index
        // can address every point in any series. (Different series are
        // allowed to have different lengths — e.g. val_* reports lag the
        // train reports.) The driver series also supplies the x-axis step
        // numbers when present.
        let driver = series.max(by: { $0.values.count < $1.values.count })
        let sampleCount = driver?.values.count ?? 0
        let xDomain = xStepDomain()

        return HStack(spacing: 8) {
            ZStack(alignment: .topLeading) {
                // Faint baseline grid
                Path { path in
                    let step = plotHeight / 4
                    for i in 1...3 {
                        let y = step * CGFloat(i)
                        path.move(to: CGPoint(x: 0, y: y))
                        path.addLine(to: CGPoint(x: plotWidth, y: y))
                    }
                }
                .stroke(Color.white.opacity(0.05), lineWidth: 0.5)

                ForEach(series) { s in
                    seriesPath(s, in: plotRect, xDomain: xDomain)
                }

                // Hover overlay: vertical guide, per-series dots, and a
                // floating callout. Drawn on top of the lines so it sits
                // above them. The hit area is the whole plot region.
                if let hoverStep, sampleCount > 0 {
                    hoverOverlay(
                        hoverStep: hoverStep,
                        driverSteps: driver?.xSteps,
                        plotRect: plotRect,
                        xDomain: xDomain
                    )
                }
            }
            .frame(width: plotWidth, height: plotHeight)
            .contentShape(Rectangle())
            .onContinuousHover { phase in
                switch phase {
                case .active(let point):
                    let xRatio = max(0, min(1, Double(point.x / plotWidth)))
                    let rawStep = Double(xDomain.lowerBound) + xRatio * Double(max(xDomain.upperBound - xDomain.lowerBound, 1))
                    let nextStep = nearestStep(to: Int(rawStep.rounded()), in: driver)
                    if hoverStep != nextStep { hoverStep = nextStep }
                case .ended:
                    if hoverStep != nil { hoverStep = nil }
                }
            }

            legend
                .frame(width: legendWidth, alignment: .topLeading)
        }
    }

    /// Width reserved for the legend on the right of the chart. Tuned
    /// for ~5-6 series; long labels truncate.
    private var legendWidth: CGFloat { 110 }

    @ViewBuilder
    private func seriesPath(_ s: Series, in rect: CGRect, xDomain: ClosedRange<Int>) -> some View {
        let values = s.values
        if values.count < 2 {
            EmptyView()
        } else {
            let minV = values.min() ?? 0
            let maxV = values.max() ?? 1
            let span = max(maxV - minV, .ulpOfOne)

            Path { path in
                for index in values.indices {
                    let x = xPosition(for: s, at: index, in: rect, xDomain: xDomain)
                    let yRatio = (values[index] - minV) / span
                    let y = rect.height - (rect.height * CGFloat(yRatio))
                    if index == values.startIndex {
                        path.move(to: CGPoint(x: x, y: y))
                    } else {
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
            }
            .stroke(
                s.color,
                style: StrokeStyle(
                    lineWidth: 1.8,
                    lineCap: .round,
                    lineJoin: .round,
                    dash: s.isDashed ? [4, 3] : []
                )
            )
        }
    }

    /// Per-series point in normalized plot space.
    private func point(for s: Series, at index: Int, in rect: CGRect, xDomain: ClosedRange<Int>) -> CGPoint? {
        let values = s.values
        guard index >= 0, index < values.count, values.count >= 2 else { return nil }
        let minV = values.min() ?? 0
        let maxV = values.max() ?? 1
        let span = max(maxV - minV, .ulpOfOne)
        let x = xPosition(for: s, at: index, in: rect, xDomain: xDomain)
        let yRatio = (values[index] - minV) / span
        let y = rect.height - (rect.height * CGFloat(yRatio))
        return CGPoint(x: x, y: y)
    }

    @ViewBuilder
    private func hoverOverlay(hoverStep: Int, driverSteps: [Int]?, plotRect: CGRect, xDomain: ClosedRange<Int>) -> some View {
        // Snap to the x position of the driver series. The vertical guide
        // is drawn as a single thin line in front of the data.
        let driver = series.max(by: { $0.values.count < $1.values.count })
        let hoverIndex = nearestIndex(in: driver, toStep: hoverStep) ?? 0
        let x = xPosition(for: driver, at: hoverIndex, in: plotRect, xDomain: xDomain)
        let stepLabel = xLabel(for: hoverStep, driverSteps: driverSteps)

        Path { path in
            path.move(to: CGPoint(x: x, y: 0))
            path.addLine(to: CGPoint(x: x, y: plotRect.height))
        }
        .stroke(Color.white.opacity(0.35), style: StrokeStyle(lineWidth: 0.8, dash: [3, 3]))

        ForEach(series) { s in
            if let seriesIndex = nearestIndex(in: s, toStep: hoverStep),
               let p = point(for: s, at: seriesIndex, in: plotRect, xDomain: xDomain) {
                Circle()
                    .fill(s.color)
                    .frame(width: 7, height: 7)
                    .overlay(
                        Circle().stroke(Color.white.opacity(0.85), lineWidth: 1)
                    )
                    .position(x: p.x, y: p.y)
            }
        }

        // Floating callout: pin to the right of the cursor unless the
        // cursor is in the rightmost 30% of the plot, in which case pin
        // to the left so it never overflows the chart. The callout lists
        // the step label on top and one row per series with its value
        // nearest reported step in each series. This keeps sparse
        // validation samples visible when the training series has many
        // more points.
        let calloutWidth: CGFloat = min(180, plotRect.width * 0.45)
        let calloutHeight = CGFloat(20 + 16 * series.count)
        let calloutX: CGFloat = (x + calloutWidth + 8 <= plotRect.width)
            ? x + 8
            : max(4, x - calloutWidth - 8)
        let calloutY: CGFloat = max(2, min(plotRect.height - calloutHeight - 2, 2))

        VStack(alignment: .leading, spacing: 4) {
            Text(stepLabel)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
            ForEach(series) { s in
                if let seriesIndex = nearestIndex(in: s, toStep: hoverStep),
                   s.values.indices.contains(seriesIndex) {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(s.color)
                            .frame(width: 6, height: 6)
                        Text(s.label)
                            .font(.caption2)
                            .lineLimit(1)
                        Spacer(minLength: 4)
                        Text(formatValue(s.values[seriesIndex]))
                            .font(.caption2.monospacedDigit())
                            .lineLimit(1)
                    }
                }
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .frame(width: calloutWidth, height: calloutHeight, alignment: .topLeading)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(.black.opacity(0.55))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .stroke(Color.white.opacity(0.10), lineWidth: 0.5)
        )
        .position(x: calloutX + calloutWidth / 2, y: calloutY + calloutHeight / 2)
        .allowsHitTesting(false)
    }

    /// Label shown at the top of the hover callout. Prefers the trainer's
    /// absolute step number from the driver series, otherwise falls back
    /// to the raw sample index.
    private func xLabel(for step: Int, driverSteps: [Int]?) -> String {
        if driverSteps != nil {
            return "step \(step)"
        }
        return "sample \(step)"
    }

    private func xStep(for index: Int, in series: Series) -> Int {
        if let steps = series.xSteps, steps.indices.contains(index) {
            return steps[index]
        }
        return index
    }

    private func xPosition(for series: Series?, at index: Int, in rect: CGRect, xDomain: ClosedRange<Int>) -> CGFloat {
        guard let series else { return CGFloat(index) }
        let step = xStep(for: index, in: series)
        let span = max(xDomain.upperBound - xDomain.lowerBound, 1)
        let ratio = Double(step - xDomain.lowerBound) / Double(span)
        return rect.width * CGFloat(max(0, min(1, ratio)))
    }

    private func xStepDomain() -> ClosedRange<Int> {
        let steps = series.flatMap { s in
            s.values.indices.map { xStep(for: $0, in: s) }
        }
        guard let minStep = steps.min(), let maxStep = steps.max() else {
            return 0...1
        }
        return minStep...max(maxStep, minStep + 1)
    }

    private func nearestStep(to step: Int, in series: Series?) -> Int {
        guard let index = nearestIndex(in: series, toStep: step), let series else {
            return step
        }
        return xStep(for: index, in: series)
    }

    private func nearestIndex(in series: Series?, toStep step: Int) -> Int? {
        guard let series, !series.values.isEmpty else { return nil }
        guard let steps = series.xSteps, !steps.isEmpty else {
            return min(max(step, 0), series.values.count - 1)
        }

        let usableCount = min(steps.count, series.values.count)
        guard usableCount > 0 else { return nil }

        var bestIndex = 0
        var bestDistance = abs(steps[0] - step)
        for index in 1..<usableCount {
            let distance = abs(steps[index] - step)
            if distance < bestDistance {
                bestDistance = distance
                bestIndex = index
            }
        }
        return bestIndex
    }

    private var legend: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(series) { s in
                HStack(spacing: 6) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(s.color)
                        .frame(width: 12, height: 3)
                    VStack(alignment: .leading, spacing: 0) {
                        Text(s.label)
                            .font(.caption2)
                            .lineLimit(1)
                        if let v = s.latest {
                            Text(formatValue(v))
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                }
            }
            if let yLabel {
                Spacer(minLength: 0)
                Text(yLabel)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func formatValue(_ v: Double) -> String {
        if abs(v) >= 1000 || (abs(v) < 0.01 && v != 0) {
            return String(format: "%.2e", v)
        }
        return String(format: "%.3f", v)
    }
}

// MARK: - ChartCard
// A title + subtitle + content shell used everywhere a metrics card
// is shown. The liquid-glass background keeps the chart blending
// into the rest of the app. Promoted from `LiveMetricsView.swift` so
// the Runs page's detail sheet can reuse the exact same look.

struct ChartCard<Content: View>: View {
    let title: String
    let subtitle: String
    @ViewBuilder var content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.headline)
            Text(subtitle)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
            content()
                .padding(.top, 6)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .liquidGlass(cornerRadius: 10)
    }
}

// MARK: - MiniSparkline
// A self-contained, axis-free mini chart used on the Runs page cards
// to give a visual "shape" hint for each run. Renders a single
// polyline normalised to its own min/max, in the given colour. Valid
// is overlaid as a dashed line when present.

struct MiniSparkline: View {
    var trainValues: [Double]
    var valValues: [Double] = []
    var trainColor: Color = .orange
    var valColor: Color = .blue
    var lineWidth: CGFloat = 1.5

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                trainPath(in: proxy.size)
                    .stroke(trainColor, style: StrokeStyle(lineWidth: lineWidth, lineCap: .round, lineJoin: .round))
                if !valValues.isEmpty {
                    valPath(in: proxy.size)
                        .stroke(valColor, style: StrokeStyle(lineWidth: lineWidth, lineCap: .round, lineJoin: .round, dash: [3, 2]))
                }
            }
        }
    }

    private func trainPath(in size: CGSize) -> Path {
        path(for: trainValues, in: size, useTrainRange: true)
    }

    private func valPath(in size: CGSize) -> Path {
        path(for: valValues, in: size, useTrainRange: false)
    }

    /// Draw a polyline through the supplied values. The y-axis is
    /// normalised against the *train* series so the two lines share a
    /// scale — that way a spike in the train loss shows up as a
    /// corresponding dip/rise in the val line at the same point on
    /// the y axis.
    private func path(for values: [Double], in size: CGSize, useTrainRange: Bool) -> Path {
        guard values.count >= 2 else { return Path() }
        let reference = useTrainRange ? trainValues : values
        guard let minV = reference.min(), let maxV = reference.max() else { return Path() }
        let span = max(maxV - minV, .ulpOfOne)
        var path = Path()
        for (index, value) in values.enumerated() {
            let x = size.width * CGFloat(index) / CGFloat(max(values.count - 1, 1))
            let yRatio = (value - minV) / span
            let y = size.height - (size.height * CGFloat(yRatio))
            if index == 0 {
                path.move(to: CGPoint(x: x, y: y))
            } else {
                path.addLine(to: CGPoint(x: x, y: y))
            }
        }
        return path
    }
}

private extension CGFloat {
    func clamped01() -> CGFloat { Swift.min(Swift.max(self, 0), 1) }
}
