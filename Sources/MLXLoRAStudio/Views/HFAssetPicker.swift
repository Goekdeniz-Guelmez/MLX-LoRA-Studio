import SwiftUI
import AppKit

struct HFAssetPicker: View {
    @Binding var text: String
    let kind: HFCachedAsset.Kind
    let placeholder: String
    var footer: String? = nil

    @Environment(AppStore.self) private var store

    @State private var showingAddHFSheet = false
    @State private var newHFInput: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: kind == .model ? "cpu" : "tray")
                    .foregroundStyle(.secondary)
                TextField(placeholder, text: $text)
                    .textFieldStyle(.roundedBorder)
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
                Button {
                    Task {
                        await store.refreshHFCache()
                        if kind == .dataset {
                            await store.refreshLocalRunOutputs()
                        }
                    }
                } label: {
                    if store.isScanningHFCache {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: "arrow.clockwise")
                    }
                }
                .buttonStyle(.borderless)
                .help("Re-scan the local Hugging Face cache")
            }
            if let footer {
                HStack(spacing: 6) {
                    Text(footer)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    if store.huggingFaceTokenIsSet {
                        Label("HF key set", systemImage: "key.fill")
                            .labelStyle(.titleAndIcon)
                            .font(.caption2)
                            .foregroundStyle(.green)
                            .help("A Hugging Face token is stored in the Keychain. Gated and private repos will authenticate automatically.")
                    } else {
                        Label("No HF key", systemImage: "key")
                            .labelStyle(.titleAndIcon)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                            .help("Add a token in Settings → Hugging Face to load gated or private repos.")
                    }
                }
            }
        }
        .sheet(isPresented: $showingAddHFSheet) {
            AddHFAssetSheet(
                kind: kind,
                initial: newHFInput,
                onCommit: { id in
                    let clean = id.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !clean.isEmpty else { return }
                    store.addCustomHFAsset(clean, kind: kind)
                    text = clean.contains("/")
                        ? clean
                        : clean.replacingOccurrences(of: "--", with: "/")
                    newHFInput = ""
                    showingAddHFSheet = false
                },
                onCancel: {
                    newHFInput = ""
                    showingAddHFSheet = false
                },
            )
        }
    }

    // MARK: Menu

    @ViewBuilder
    private var menuContent: some View {
        let cached = kind == .model ? store.cachedModels : store.cachedDatasets
        let custom = kind == .model ? store.customModelPaths : store.customDatasetPaths
        let syntheticOutputs = store.syntheticRunOutputs

        Section("In local HF cache (\(cached.count))") {
            if cached.isEmpty {
                Text("Nothing downloaded yet").foregroundStyle(.secondary)
            } else {
                ForEach(cached) { asset in
                    Button {
                        text = asset.hfID
                    } label: {
                        Label(asset.hfID, systemImage: kind == .model ? "cpu" : "tray")
                    }
                }
            }
        }

        if kind == .dataset {
            Section("Synthetic runs (\(syntheticOutputs.count))") {
                if syntheticOutputs.isEmpty {
                    Text("No generated-data folders yet").foregroundStyle(.secondary)
                } else {
                    ForEach(syntheticOutputs) { output in
                        Button {
                            text = output.path
                        } label: {
                            Label(output.name, systemImage: "sparkles")
                        }
                    }
                }
            }
        }

        if !custom.isEmpty {
            Section("My custom paths (\(custom.count))") {
                ForEach(custom, id: \.self) { path in
                    Button {
                        text = path
                    } label: {
                        HStack {
                            Label(path, systemImage: "bookmark")
                            Spacer()
                            Button {
                                store.removeCustomPath(path, kind: kind)
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                            }
                            .buttonStyle(.borderless)
                        }
                    }
                }
            }
        }

        Section {
            Button {
                newHFInput = ""
                showingAddHFSheet = true
            } label: {
                Label("Add HF repo…", systemImage: "plus.circle")
            }
            Button { pickLocalPathWithPanel() } label: {
                Label("Browse local path…", systemImage: "folder.badge.plus")
            }
        }
    }

    // MARK: Local path picker
    private func pickLocalPathWithPanel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = (kind == .dataset)
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        panel.message = kind == .model
            ? "Pick a local model folder (the directory that contains config.json)."
            : "Pick a local dataset folder or file."
        panel.prompt = "Use Path"
        if panel.runModal() == .OK, let url = panel.url {
            store.addCustomLocalPath(url.path, kind: kind)
            text = url.path
        }
    }
}

// MARK: - "Add HF repo…" sheet
private struct AddHFAssetSheet: View {
    let kind: HFCachedAsset.Kind
    @State var initial: String
    let onCommit: (String) -> Void
    let onCancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(kind == .model ? "Add Hugging Face model" : "Add Hugging Face dataset")
                .font(.headline)
            Text("Paste a repo id from huggingface.co. Both `owner/name` and `owner--name` are accepted.")
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            TextField(kind == .model ? "owner/model-name" : "owner/dataset-name", text: $initial)
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 360)
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { onCancel() }
                Button("Add") {
                    onCommit(initial)
                }
                .keyboardShortcut(.defaultAction)
                .disabled(initial.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
        .frame(minWidth: 400)
    }
}
