import SwiftUI

/// Settings section that lets the user pick which Python interpreter the app
/// uses for jobs, refresh the list of discovered envs, and create a new
/// venv or conda env without leaving the UI.
struct PythonPickerSection: View {
    @Bindable var store: AppStore
    @State private var isShowingCreateSheet = false
    @State private var selection: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Picker("Interpreter", selection: $selection) {
                Text(store.pythonEnvironments.isEmpty && !store.isScanningPythons
                     ? "No interpreters found"
                     : "Select interpreter")
                    .tag("")
                ForEach(store.pythonEnvironments) { env in
                    Text("\(env.label) — Python \(env.version)")
                        .tag(env.path)
                }
                Text("Custom path…").tag("__custom__")
            }
            .onChange(of: selection) { _, newValue in
                if newValue == "__custom__" {
                    if !store.customPythonPath.isEmpty {
                        store.selectCustomPython(store.customPythonPath)
                    }
                } else if !newValue.isEmpty, let env = store.pythonEnvironments.first(where: { $0.path == newValue }) {
                    store.selectPython(env)
                }
            }
            .onAppear { syncSelectionFromStore() }
            .onChange(of: store.pythonEnvironments.map(\.id)) { _, _ in
                syncSelectionFromStore()
            }

            HStack(spacing: 10) {
                Button {
                    Task { await store.refreshPythonEnvironments() }
                } label: {
                    if store.isScanningPythons {
                        ProgressView().controlSize(.small)
                    } else {
                        Label("Refresh", systemImage: "arrow.clockwise")
                    }
                }
                .disabled(store.isScanningPythons)

                Button {
                    isShowingCreateSheet = true
                } label: {
                    Label("Create new environment…", systemImage: "plus.circle")
                }
                .disabled(store.pythonEnvironments.isEmpty)
            }

            if selection == "__custom__" {
                TextField("/path/to/python", text: Binding(
                    get: { store.customPythonPath },
                    set: { store.selectCustomPython($0) }
                ))
                .textFieldStyle(.roundedBorder)
                .font(.system(.body, design: .monospaced))
            }

            statusLine
        }
        .sheet(isPresented: $isShowingCreateSheet) {
            CreatePythonEnvSheet(
                store: store,
                baseInterpreter: store.pythonEnvironments.first ?? nil,
                onCreate: { name, kind, base in
                    isShowingCreateSheet = false
                    Task { await store.createPythonEnv(name: name, kind: kind, base: base) }
                }
            )
        }
    }

    @ViewBuilder
    private var statusLine: some View {
        if let env = store.selectedPythonEnvironment {
            HStack(spacing: 6) {
                Text("v\(env.version) · \(env.label)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                activationBadge
            }
        } else if !store.customPythonPath.isEmpty {
            HStack(spacing: 6) {
                Text("Custom path")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                activationBadge
            }
        } else {
            Text("No interpreter selected")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private var activationBadge: some View {
        switch store.pythonActivationOK {
        case .some(true):
            EmptyView()
        case .some(false):
            Label("mlx_lm_lora not installed", systemImage: "exclamationmark.triangle.fill")
                .labelStyle(.titleAndIcon)
                .font(.caption)
                .foregroundStyle(.orange)
        case .none:
            ProgressView().controlSize(.mini)
        }
    }

    private func syncSelectionFromStore() {
        let target = store.pythonExecutable
        if store.pythonEnvironments.contains(where: { $0.path == target }) {
            selection = target
        } else if !store.customPythonPath.isEmpty && target == store.customPythonPath {
            selection = "__custom__"
        } else if !target.isEmpty {
            // Active path isn't discovered yet — drop into custom so the user sees it.
            selection = "__custom__"
        } else {
            selection = store.pythonEnvironments.first?.path ?? ""
        }
    }
}

// MARK: - Create sheet

private struct CreatePythonEnvSheet: View {
    @Bindable var store: AppStore
    let baseInterpreter: PythonEnvironment?
    let onCreate: (String, PythonEnvironment.Kind, PythonEnvironment) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var name: String = "mlx-lora-studio"
    @State private var kind: PythonEnvironment.Kind = .venv
    @State private var basePath: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Create Python Environment")
                .font(.title3.bold())

            Form {
                TextField("Name", text: $name)
                Picker("Kind", selection: $kind) {
                    Text("Venv").tag(PythonEnvironment.Kind.venv)
                    Text("Conda").tag(PythonEnvironment.Kind.conda)
                }
                .pickerStyle(.segmented)

                Picker("Base interpreter", selection: $basePath) {
                    Text("Select base interpreter").tag("")
                    ForEach(store.pythonEnvironments) { env in
                        Text("\(env.label) — Python \(env.version)")
                            .tag(env.path)
                    }
                }

                if kind == .venv {
                    LabeledContent("Location") {
                        Text(locationPreview)
                            .font(.system(.callout, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
            }
            .formStyle(.grouped)

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Create") {
                    if let base = baseEnv {
                        onCreate(sanitizedName, kind, base)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(sanitizedName.isEmpty || baseEnv == nil || store.isCreatingPythonEnv)
            }
        }
        .padding(20)
        .frame(width: 460)
        .onAppear {
            if basePath.isEmpty, let first = baseInterpreter?.path ?? store.pythonEnvironments.first?.path {
                basePath = first
            }
        }
    }

    private var baseEnv: PythonEnvironment? {
        store.pythonEnvironments.first { $0.path == basePath }
    }

    private var sanitizedName: String {
        name.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var locationPreview: String {
        let parent = PythonEnvironmentProvisioner.defaultVenvParent()
        return parent
            .appending(path: sanitizedName, directoryHint: .isDirectory)
            .appending(path: "bin/python")
            .path
    }
}
