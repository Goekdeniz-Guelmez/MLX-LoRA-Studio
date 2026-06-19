import SwiftUI

/// A page that visualises every metric the active trainer has emitted
/// so far. Discovery is dynamic: cards appear as the relevant keys
/// start landing in `runner.metrics`, and disappear (well, become
/// empty placeholders) if a future step stops producing them. The
/// classic single-series loss chart is kept full-width at the top
/// because it's the most-read signal during training.
struct LiveMetricsView: View {
    @Bindable var runner: PythonJobRunner

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header
                summaryTiles
                lossCard
                autoDiscoveredGrid
                metricsTableCard
                recentRawLinesCard
            }
            .padding(24)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .liquidGlass(cornerRadius: 18)
        .padding(16)
    }

    // MARK: - Header

    private var header: some View {
        HeaderView(
            title: "Live Metrics",
            subtitle: runner.metrics.isEmpty
                ? "Waiting for trainer output…"
                : "Auto-discovered from the active trainer",
            symbol: "chart.line.uptrend.xyaxis"
        )
    }

    // MARK: - Summary tiles

    private var summaryTiles: some View {
        HStack(spacing: 10) {
            StatTile(title: "Step", value: latestStep)
            StatTile(title: "Loss", value: latestLoss)
            StatTile(title: "Speed", value: latestItPerSec)
            StatTile(title: "Tokens/s", value: latestTokPerSec)
            StatTile(title: "Memory", value: latestMemory)
            StatTile(title: "LR", value: latestLearningRate)
        }
    }

    private var latest: TrainingMetric? { runner.metrics.last }
    private var latestStep: String { latest.map { "Iter \($0.step)" } ?? "-" }
    private var latestLoss: String { latest?.loss.map { String(format: "%.3f", $0) } ?? "-" }
    private var latestMemory: String { latest?.memoryGB.map { String(format: "%.2f GB", $0) } ?? "-" }
    private var latestItPerSec: String { latest?.values["it_s"].map { String(format: "%.2f it/s", $0) } ?? "-" }
    private var latestTokPerSec: String { latest?.values["tok_s"].map { String(format: "%.0f", $0) } ?? "-" }
    private var latestLearningRate: String { latest?.learningRate.map { String(format: "%.2e", $0) } ?? "-" }

    // MARK: - Loss chart (full width)

    private var lossCard: some View {
        // Walk the metrics once to keep train + val values paired with their
        // step numbers. Validation reports lag training and skip samples,
        // so each series carries its own xSteps for the hover callout.
        var trainValues: [Double] = []
        var trainSteps: [Int] = []
        var valValues: [Double] = []
        var valSteps: [Int] = []
        var testValues: [Double] = []
        var testSteps: [Int] = []
        for metric in runner.metrics {
            if let loss = metric.loss {
                trainValues.append(loss)
                trainSteps.append(metric.step)
            }
            if let valLoss = metric.validationLoss {
                valValues.append(valLoss)
                valSteps.append(metric.step)
            }
            if let testLoss = metric.testLoss {
                testValues.append(testLoss)
                testSteps.append(metric.step)
            }
        }

        var series: [MultiLineChart.Series] = [
            MultiLineChart.Series(
                id: "loss",
                label: "Train",
                color: .orange,
                values: trainValues,
                xSteps: trainSteps
            )
        ]
        if !valValues.isEmpty {
            series.append(
                MultiLineChart.Series(
                    id: "val_loss",
                    label: "Validation",
                    color: .blue,
                    values: valValues,
                    isDashed: true,
                    xSteps: valSteps
                )
            )
        }
        if !testValues.isEmpty {
            series.append(
                MultiLineChart.Series(
                    id: "test_loss",
                    label: "Test",
                    color: .green,
                    values: testValues,
                    isDashed: true,
                    xSteps: testSteps
                )
            )
        }

        let subtitle = valValues.isEmpty && testValues.isEmpty
            ? "Training loss across iterations. The single most-read signal."
            : "Training (solid), validation and test (dashed) loss across iterations."

        return ChartCard(title: "Loss", subtitle: subtitle) {
            MultiLineChart(series: series, yLabel: "loss")
        }
        .frame(minHeight: 180)
    }

    // MARK: - Auto-discovered grid

    @ViewBuilder
    private var autoDiscoveredGrid: some View {
        let groups = MetricCardGroup.compute(from: runner.metrics)
        if groups.isEmpty {
            EmptyView()
        } else {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 320, maximum: 540), spacing: 16)], spacing: 16) {
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
    }

    // MARK: - Metrics table

    private var metricsTableCard: some View {
        let rows = MetricTableRow.compute(from: runner.metrics)
        return VStack(alignment: .leading, spacing: 10) {
            SectionTitle("All Metrics")
            Text("Every key the trainer has reported. Validation and test keys keep their split prefix.")
                .font(.callout)
                .foregroundStyle(.secondary)
            if rows.isEmpty {
                Text("No metrics yet.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 12)
            } else {
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
        }
        .formBlock()
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
                if row.isValidation {
                    Image(systemName: "checkmark.seal.fill")
                        .font(.caption2)
                        .foregroundStyle(.blue.opacity(0.8))
                } else {
                    Image(systemName: "bolt.fill")
                        .font(.caption2)
                        .foregroundStyle(.orange.opacity(0.8))
                }
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

    // MARK: - Recent raw lines

    private var recentRawLinesCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionTitle("Recent Reports")
            Text("Last few trainer reports verbatim. Useful when a chart spike needs context.")
                .font(.callout)
                .foregroundStyle(.secondary)
            let recents = Array(runner.metrics.suffix(5).reversed())
            if recents.isEmpty {
                Text("No reports yet.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 8)
            } else {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(recents) { metric in
                        rawLineBlock(metric)
                    }
                }
                .padding(10)
                .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
            }
        }
        .formBlock()
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

// MARK: - Chart card wrapper


// MARK: - Auto-discovery

/// One chart card, derived from a logical group of metric keys. The
/// grouping rules are static (preference rewards, KL, clipping, etc.)
/// because the trainers always emit those names — but which groups
/// actually render depends on which keys have non-empty data in
/// `runner.metrics`.
struct MetricCardGroup: Identifiable {
    let id: String
    let title: String
    let subtitle: String
    let series: [MultiLineChart.Series]

    static func compute(from metrics: [TrainingMetric]) -> [MetricCardGroup] {
        guard !metrics.isEmpty else { return [] }
        var groups: [MetricCardGroup] = []

        // Preference rewards (DPO/CPO/ORPO/PPO/Online-DPO/XPO)
        let pref = preferenceRewards(from: metrics)
        if !pref.isEmpty {
            groups.append(MetricCardGroup(
                id: "preference",
                title: "Preference Rewards",
                subtitle: "Chosen vs rejected reward on training (solid) and validation (dashed) batches.",
                series: pref
            ))
        }

        // RLHF Reinforce: rewards + kl_penalty + advantages
        let rlhf = rlhfMetrics(from: metrics)
        if !rlhf.isEmpty {
            groups.append(MetricCardGroup(
                id: "rlhf",
                title: "RLHF",
                subtitle: "Per-step rewards, KL penalty, and advantages from the policy-gradient loop.",
                series: rlhf
            ))
        }

        // GRPO reward distribution: total_rewards + grouped_rewards
        let dist = rewardDistribution(from: metrics)
        if !dist.isEmpty {
            groups.append(MetricCardGroup(
                id: "reward-dist",
                title: "Reward Distribution",
                subtitle: "Total and per-group reward mean and std from the GRPO rollout.",
                series: dist
            ))
        }

        // KL divergence (GRPO standalone)
        if let klSeries = singleSeries(key: "kl", label: "KL", color: .teal, from: metrics) {
            groups.append(MetricCardGroup(
                id: "kl",
                title: "KL Divergence",
                subtitle: "Per-step KL between the current and reference policy.",
                series: [klSeries]
            ))
        }

        // PPO clipping
        let clipping = clippingRatios(from: metrics)
        if !clipping.isEmpty {
            groups.append(MetricCardGroup(
                id: "clipping",
                title: "PPO Clipping",
                subtitle: "Fraction of tokens clipped at the low / high / total ratio.",
                series: clipping
            ))
        }

        // Generation tokens (GRPO)
        let gen = generation(from: metrics)
        if !gen.isEmpty {
            groups.append(MetricCardGroup(
                id: "generation",
                title: "Generation",
                subtitle: "Token counts per rollout and the share that hit the max-length cap.",
                series: gen
            ))
        }

        // Per-reward functions (GRPO)
        let perReward = perRewardFunctions(from: metrics)
        for group in perReward {
            groups.append(group)
        }

        // Throughput (it/s, tok/s) — always useful, render even with 1 point
        let throughput = throughput(from: metrics)
        if !throughput.isEmpty {
            groups.append(MetricCardGroup(
                id: "throughput",
                title: "Throughput",
                subtitle: "Iterations and tokens per second as reported by the trainer.",
                series: throughput
            ))
        }

        return groups
    }

    // MARK: - Series builders

    private static func preferenceRewards(from metrics: [TrainingMetric]) -> [MultiLineChart.Series] {
        var series: [MultiLineChart.Series] = []
        if let s = singleSeries(key: "chosen_r", label: "Chosen (train)", color: .green, from: metrics, isValidation: false) {
            series.append(s)
        }
        if let s = singleSeries(key: "rejected_r", label: "Rejected (train)", color: .red, from: metrics, isValidation: false) {
            series.append(s)
        }
        if let s = singleSeries(key: "val_chosen_r", label: "Chosen (val)", color: .green, from: metrics, isValidation: true) {
            series.append(s)
        }
        if let s = singleSeries(key: "val_rejected_r", label: "Rejected (val)", color: .red, from: metrics, isValidation: true) {
            series.append(s)
        }
        return series
    }

    private static func rlhfMetrics(from metrics: [TrainingMetric]) -> [MultiLineChart.Series] {
        var series: [MultiLineChart.Series] = []
        if let s = singleSeries(key: "rewards", label: "Rewards", color: .green, from: metrics) { series.append(s) }
        if let s = singleSeries(key: "kl_penalty", label: "KL Penalty", color: .teal, from: metrics) { series.append(s) }
        if let s = singleSeries(key: "advantages", label: "Advantages", color: .purple, from: metrics) { series.append(s) }
        return series
    }

    private static func rewardDistribution(from metrics: [TrainingMetric]) -> [MultiLineChart.Series] {
        var series: [MultiLineChart.Series] = []
        if let s = singleSeries(key: "total_rewards_mean", label: "Total μ", color: .green, from: metrics) { series.append(s) }
        if let s = singleSeries(key: "total_rewards_std", label: "Total σ", color: .green.opacity(0.55), from: metrics) { series.append(s) }
        if let s = singleSeries(key: "grouped_rewards_mean", label: "Group μ", color: .orange, from: metrics) { series.append(s) }
        if let s = singleSeries(key: "grouped_rewards_std", label: "Group σ", color: .orange.opacity(0.55), from: metrics) { series.append(s) }
        return series
    }

    private static func clippingRatios(from metrics: [TrainingMetric]) -> [MultiLineChart.Series] {
        var series: [MultiLineChart.Series] = []
        if let s = singleSeries(key: "clip_ratio_low", label: "Low", color: .blue, from: metrics) { series.append(s) }
        if let s = singleSeries(key: "clip_ratio_high", label: "High", color: .indigo, from: metrics) { series.append(s) }
        if let s = singleSeries(key: "clip_ratio_total", label: "Total", color: .purple, from: metrics) { series.append(s) }
        return series
    }

    private static func generation(from metrics: [TrainingMetric]) -> [MultiLineChart.Series] {
        var series: [MultiLineChart.Series] = []
        if let s = singleSeries(key: "avg_generated_tokens", label: "Avg tokens", color: .purple, from: metrics) { series.append(s) }
        if let s = singleSeries(key: "min_generated_tokens", label: "Min tokens", color: .indigo, from: metrics) { series.append(s) }
        if let s = singleSeries(key: "max_generated_tokens", label: "Max tokens", color: .pink, from: metrics) { series.append(s) }
        if let s = singleSeries(key: "hit_max_tokens_ratio", label: "Hit limit", color: .orange, from: metrics) { series.append(s) }
        return series
    }

    private static func throughput(from metrics: [TrainingMetric]) -> [MultiLineChart.Series] {
        var series: [MultiLineChart.Series] = []
        if let s = singleSeries(key: "it_s", label: "it/s", color: .cyan, from: metrics) { series.append(s) }
        if let s = singleSeries(key: "tok_s", label: "tok/s", color: .mint, from: metrics) { series.append(s) }
        return series
    }

    /// Build a card per reward function defined in the GRPO config.
    /// Each function emits `reward_<name>_mean`, `..._std`, `..._coverage`.
    private static func perRewardFunctions(from metrics: [TrainingMetric]) -> [MetricCardGroup] {
        // Find every distinct reward-function name by scanning for
        // `reward_<name>_mean` keys. We never want to assume a fixed
        // list of reward functions — the user configures them.
        var names: [String] = []
        var seen: Set<String> = []
        for metric in metrics {
            for key in metric.values.keys {
                guard key.hasPrefix("reward_"), key.hasSuffix("_mean") else { continue }
                let middle = key
                    .dropFirst("reward_".count)
                    .dropLast("_mean".count)
                let name = String(middle)
                if !seen.contains(name) {
                    seen.insert(name)
                    names.append(name)
                }
            }
        }
        return names.map { name in
            var series: [MultiLineChart.Series] = []
            if let s = singleSeries(key: "reward_\(name)_mean", label: "μ", color: .green, from: metrics) { series.append(s) }
            if let s = singleSeries(key: "reward_\(name)_std", label: "σ", color: .yellow, from: metrics) { series.append(s) }
            if let s = singleSeries(key: "reward_\(name)_coverage", label: "Coverage", color: .blue, from: metrics) { series.append(s) }
            return MetricCardGroup(
                id: "reward-\(name)",
                title: "Reward · \(humanise(name))",
                subtitle: "Per-reward mean, std, and coverage across the rollout.",
                series: series
            )
        }
    }

    // MARK: - Helpers

    /// Pull a single key out of the metrics array, dropping `nil`s. Returns
    /// `nil` if there are no values for the key, so callers can decide
    /// whether to include the resulting series in a card.
    private static func singleSeries(
        key: String,
        label: String,
        color: Color,
        from metrics: [TrainingMetric],
        isValidation: Bool = false
    ) -> MultiLineChart.Series? {
        // Walk once and pick up (value, step) pairs in order. Validation
        // series tend to lag training and skip samples, so a compactMap on
        // values alone loses the step mapping. The chart's hover overlay
        // shows "step N" from the i-th entry of `xSteps`.
        var values: [Double] = []
        var xSteps: [Int] = []
        values.reserveCapacity(metrics.count)
        xSteps.reserveCapacity(metrics.count)
        for metric in metrics {
            if let v = metric.values[key] {
                values.append(v)
                xSteps.append(metric.step)
            }
        }
        guard !values.isEmpty else { return nil }
        return MultiLineChart.Series(
            id: key,
            label: label,
            color: color,
            values: values,
            isDashed: isValidation,
            xSteps: xSteps
        )
    }

    private static func humanise(_ key: String) -> String {
        // `helpfulness_v2` → `Helpfulness V2`
        key.replacingOccurrences(of: "_", with: " ")
            .capitalized
    }
}

// MARK: - Metrics table

struct MetricTableRow: Identifiable {
    let id: String
    let rawKey: String
    let displayName: String
    let isValidation: Bool
    let latest: Double
    let min: Double
    let max: Double
    let count: Int

    static func compute(from metrics: [TrainingMetric]) -> [MetricTableRow] {
        // Group values by key, keeping the order in which each key first
        // appeared. This gives a stable table layout.
        var order: [String] = []
        var buckets: [String: [Double]] = [:]
        for metric in metrics {
            for (key, value) in metric.values {
                if buckets[key] == nil {
                    order.append(key)
                    buckets[key] = []
                }
                buckets[key]?.append(value)
            }
        }
        return order.compactMap { key in
            guard let values = buckets[key], !values.isEmpty else { return nil }
            return MetricTableRow(
                id: key,
                rawKey: key,
                displayName: humanise(key),
                isValidation: key.hasPrefix("val_"),
                latest: values.last ?? 0,
                min: values.min() ?? 0,
                max: values.max() ?? 0,
                count: values.count
            )
        }
    }

    private static func humanise(_ key: String) -> String {
        // `chosen_r` → `Chosen R`, `kl_penalty` → `KL Penalty`,
        // `reward_helpfulness_mean` → `Helpfulness (mean)`.
        var working = key
        var suffix: String? = nil
        for s in ["_mean", "_std", "_coverage"] {
            if working.hasSuffix(s) {
                suffix = String(s.dropFirst())
                working = String(working.dropLast(s.count))
                break
            }
        }
        let core = working
            .replacingOccurrences(of: "_r", with: " R")
            .replacingOccurrences(of: "_s", with: "/s")
            .replacingOccurrences(of: "tok_s", with: "tok/s")
            .replacingOccurrences(of: "it_s", with: "it/s")
            .replacingOccurrences(of: "_", with: " ")
            .capitalized
        if let suffix {
            return "\(core) (\(suffix))"
        }
        return core
    }
}
