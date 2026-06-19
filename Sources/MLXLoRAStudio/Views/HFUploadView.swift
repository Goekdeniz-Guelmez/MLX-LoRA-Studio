import AppKit
import SwiftUI

struct HFUploadView: View {
    @Bindable var store: AppStore
    @State private var manualConsoleCollapsed: Bool? = nil

    var body: some View {
        GeometryReader { proxy in
            let available = proxy.size.width
            let shouldAutoCollapse = available < 900
            let isCollapsed = effectiveCollapsed(shouldAutoCollapse: shouldAutoCollapse)
            let settingsWidth = available * 0.62
            let consoleWidth = available - settingsWidth

            HSplitView {
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        HeaderView(
                            title: "Upload to HF",
                            subtitle: store.huggingFaceTokenIsSet ? "HF token ready" : "Add an HF token in Settings first",
                            symbol: "arrow.up.circle"
                        )

                        UploadActionStrip(store: store)
                        ModelUploadSection(store: store)
                        SyntheticDatasetUploadSection(store: store)
                        HubSettingsSection(store: store)
                    }
                    .padding(24)
                }
                .layoutPriority(1)
                .frame(
                    minWidth: 360,
                    idealWidth: isCollapsed ? 620 : settingsWidth,
                    maxWidth: isCollapsed ? .infinity : settingsWidth
                )
                .uploadSettingsPane(cornerRadius: 18)
                .padding(16)

                LiveRunPanel(
                    runner: store.hfUploadRunner,
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
        .navigationTitle("Upload to HF")
        .task {
            await store.refreshLocalRunOutputs()
        }
    }

    private func effectiveCollapsed(shouldAutoCollapse: Bool) -> Bool {
        if let manual = manualConsoleCollapsed { return manual }
        return shouldAutoCollapse
    }
}

private struct UploadActionStrip: View {
    @Bindable var store: AppStore

    private var isRunning: Bool { store.hfUploadRunner.isRunning }
    private var canUploadModel: Bool {
        !store.hfUpload.localModelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        !store.hfUpload.modelRepo.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
    private var canUploadDataset: Bool {
        !store.hfUpload.localDatasetPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        !store.hfUpload.datasetRepo.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if isRunning {
                Button {
                    store.hfUploadRunner.stop()
                } label: {
                    Label("Cancel Upload", systemImage: "xmark.circle.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .controlSize(.large)
            } else {
                HStack(spacing: 12) {
                    Button {
                        Task { await store.startHFUpload(target: .model) }
                    } label: {
                        Label("Upload Model", systemImage: "cpu.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .disabled(!canUploadModel)

                    Button {
                        Task { await store.startHFUpload(target: .dataset) }
                    } label: {
                        Label("Upload Dataset", systemImage: "tray.and.arrow.up.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .disabled(!canUploadDataset)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }

            HStack(alignment: .top, spacing: 12) {
                InfoPill(text: "Models and datasets upload separately", symbol: "arrow.triangle.branch")
                if !store.hfUploadRunner.lastRunFolder.isEmpty {
                    InfoPill(text: "Last upload: \(store.hfUploadRunner.lastRunFolder)", symbol: "folder")
                }
            }
        }
        .formBlock()
    }
}

private struct ModelUploadSection: View {
    @Bindable var store: AppStore

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle("Model Weights")
            Picker("Upload", selection: $store.hfUpload.modelUploadKind) {
                ForEach(HFModelUploadKind.allCases) { kind in
                    Text(kind.title).tag(kind)
                }
            }
            .pickerStyle(.segmented)

            LocalRunPicker(
                title: "Training run",
                placeholder: "Choose a training run",
                selection: $store.hfUpload.localModelPath,
                outputs: store.trainingRunOutputs
            )
            CustomFolderField(title: "Custom model folder", path: $store.hfUpload.localModelPath)
            TextField("Target model repo, e.g. username/model-name", text: $store.hfUpload.modelRepo)

            Text(modelHint)
                .foregroundStyle(.secondary)
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)
        }
        .formBlock()
    }

    private var modelHint: String {
        switch store.hfUpload.modelUploadKind {
        case .adaptersOnly:
            "Uploads adapter files, adapter config, tokenizer/config metadata when present, and a generated README.md."
        case .mergedWeights:
            "Uploads the full selected model folder, including merged model weights, tokenizer/config files, and README.md."
        }
    }
}

private struct SyntheticDatasetUploadSection: View {
    @Bindable var store: AppStore

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle("Synthetic Dataset")
            LocalRunPicker(
                title: "Synthetic run",
                placeholder: "Choose a synthetic data run",
                selection: $store.hfUpload.localDatasetPath,
                outputs: store.syntheticRunOutputs
            )
            CustomFolderField(title: "Custom dataset folder", path: $store.hfUpload.localDatasetPath)
            TextField("Target dataset repo, e.g. username/dataset-name", text: $store.hfUpload.datasetRepo)
            Text("Creates or updates a Hugging Face dataset repo and writes a dataset README.md.")
                .foregroundStyle(.secondary)
                .font(.callout)
        }
        .formBlock()
    }
}

private struct HubSettingsSection: View {
    @Bindable var store: AppStore

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle("Hub Settings")
            ToggleRow("Create repos as private", isOn: $store.hfUpload.privateRepo)
            TextField("Commit message", text: $store.hfUpload.commitMessage)
        }
        .formBlock()
    }
}

private struct UploadSettingsPaneModifier: ViewModifier {
    let cornerRadius: CGFloat

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
        content
            .background {
                shape
                    .fill(.regularMaterial)
                    .shadow(color: .black.opacity(0.08), radius: 12, y: 6)
            }
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
    func uploadSettingsPane(cornerRadius: CGFloat) -> some View {
        modifier(UploadSettingsPaneModifier(cornerRadius: cornerRadius))
    }
}

private struct LocalRunPicker: View {
    let title: String
    let placeholder: String
    @Binding var selection: String
    let outputs: [LocalRunOutput]

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Picker(placeholder, selection: $selection) {
                Text(placeholder).tag("")
                ForEach(outputs) { output in
                    Text(output.name).tag(output.path)
                }
                if !selection.isEmpty && !outputs.contains(where: { $0.path == selection }) {
                    Text("Custom: \(URL(fileURLWithPath: selection).lastPathComponent)").tag(selection)
                }
            }
            .pickerStyle(.menu)
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

private struct CustomFolderField: View {
    let title: String
    @Binding var path: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            TextField(title, text: $path)
            Button {
                pickFolder()
            } label: {
                Label("Browse", systemImage: "folder")
            }
        }
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        if panel.runModal() == .OK, let url = panel.url {
            path = url.path
        }
    }
}
