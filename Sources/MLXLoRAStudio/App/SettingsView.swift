import SwiftUI

struct SettingsView: View {
    @Bindable var store: AppStore

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section("Python") {
                    PythonPickerSection(store: store)
                    TextField("mlx-lm-lora package path", text: $store.packagePath)
                    TextField("Working directory", text: $store.workingDirectory)
                }

                Section("Outputs") {
                    TextField("Run output folder", text: $store.outputRoot)
                    Text("Each job creates its own folder here. The default is ~/.mlxlorastudio/runs, and you can enter any custom folder. Leave the per-run folder name blank in Train or Synthetic Data to generate one automatically.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }

                Section("Hugging Face") {
                    HuggingFaceTokenField(store: store)
                    Text("Saved to the macOS Keychain. The token is injected as `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` into every Python job the runner launches, so gated or private repos load the same way they would from a `huggingface-cli login` session.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Section("Appearance") {
                    Toggle("Background animations", isOn: $store.decorativeAnimationsEnabled)
                    Text("Turn this off during heavy training runs to stop the decorative canvas animation and leave more GPU headroom for MLX.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }

                Section("Notifications") {
                    Toggle("Notify when jobs finish", isOn: completionNotificationsBinding)
                        .disabled(store.isUpdatingCompletionNotifications)
                    Text("Shows a macOS notification when training, synthetic data generation, or Hugging Face upload finishes. macOS may ask for notification permission the first time you turn this on.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    if !store.completionNotificationsStatus.isEmpty {
                        Text(store.completionNotificationsStatus)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                Section("Resource Guard") {
                    HStack {
                        Text("Memory limit")
                        Spacer()
                        HStack(alignment: .center, spacing: 6) {
                            TextField("", value: $store.resourceGuardMemoryPercent, format: .number)
                                .multilineTextAlignment(.trailing)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 64)
                                .labelsHidden()
                            Text("%")
                                .foregroundStyle(.secondary)
                                .frame(width: 16, alignment: .leading)
                            Stepper("Memory limit", value: $store.resourceGuardMemoryPercent, in: 10...98, step: 1)
                                .labelsHidden()
                        }
                    }
                    Text("Training runs stop before peak memory reaches this share of system RAM. Lower it for more headroom; the default is 88%.")
                        .font(.callout)
                        .foregroundStyle(.secondary)

                    Divider()

                    HStack {
                        Text("iGPU wired limit")
                        Spacer()
                        HStack(alignment: .center, spacing: 6) {
                            TextField("", value: $store.iogpuWiredLimitMB, format: .number)
                                .multilineTextAlignment(.trailing)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 96)
                                .labelsHidden()
                            Text("MB")
                                .foregroundStyle(.secondary)
                                .frame(width: 24, alignment: .leading)
                            Stepper("iGPU wired limit", value: $store.iogpuWiredLimitMB, in: 1_024...1_048_576, step: 1_024)
                                .labelsHidden()
                        }
                    }
                    HStack {
                        Text("Applies `sudo sysctl iogpu.wired_limit_mb=\(store.iogpuWiredLimitMB)` using the macOS administrator prompt. The default here is 21504 MB.")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button {
                            Task { await store.applyIOGPUWiredLimit() }
                        } label: {
                            Label(store.isApplyingIOGPUWiredLimit ? "Applying" : "Apply Limit", systemImage: "lock.shield")
                        }
                        .disabled(store.isApplyingIOGPUWiredLimit)
                    }
                    if !store.iogpuWiredLimitStatus.isEmpty {
                        Text(store.iogpuWiredLimitStatus)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                }

                Section("Onboarding") {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Product tour")
                                .font(.headline)
                            Text("Show the first-run UI tour again.")
                                .font(.callout)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button {
                            store.replayOnboarding()
                        } label: {
                            Label("Replay Tour", systemImage: "sparkles")
                        }
                    }
                }

                Section("Package Updates") {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Python backends")
                                .font(.headline)
                            Text("Updates or reinstalls MLX runtime packages, then pulls and installs the local mlx-lm-lora checkout with its requirements, including OpenAI-compatible API support.")
                                .font(.callout)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        HStack(spacing: 8) {
                            Button {
                                Task { await store.updatePackages() }
                            } label: {
                                Label("Update", systemImage: "arrow.down.circle")
                            }
                            Button {
                                Task { await store.reinstallPackages() }
                            } label: {
                                Label(store.packageRunner.isRunning ? "Running" : "Reinstall", systemImage: "arrow.triangle.2.circlepath")
                            }
                            .help("Force-reinstall MLX runtime packages and the local mlx-lm-lora package requirements in the selected Python environment.")
                        }
                        .disabled(store.packageRunner.isRunning)
                    }

                    if !store.packageRunner.logLines.isEmpty {
                        ScrollView {
                            VStack(alignment: .leading, spacing: 3) {
                                ForEach(Array(store.packageRunner.logLines.suffix(80).enumerated()), id: \.offset) { _, line in
                                    Text(ANSIText.clean(line))
                                        .font(.system(.caption, design: .monospaced))
                                        .textSelection(.enabled)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(8)
                        }
                        .frame(height: 160)
                    }
                }

                Section("Current Command") {
                    Text(store.packageRunner.currentCommand.isEmpty ? "No package command launched yet." : store.packageRunner.currentCommand)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                }
            }
            .formStyle(.grouped)
            .scrollContentBackground(.hidden)
        }
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .liquidGlass(cornerRadius: 18)
        .padding(16)
    }

    private var completionNotificationsBinding: Binding<Bool> {
        Binding(
            get: { store.completionNotificationsEnabled },
            set: { enabled in
                Task { await store.setCompletionNotificationsEnabled(enabled) }
            }
        )
    }
}

private struct HuggingFaceTokenField: View {
    @Bindable var store: AppStore

    @State private var isEditing = false
    @State private var draft = ""

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "key.fill")
                .foregroundStyle(.secondary)
            Group {
                if isEditing {
                    SecureField("hf_…", text: $draft)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit(commit)
                } else {
                    Text(placeholder)
                        .foregroundStyle(store.huggingFaceTokenIsSet ? .primary : .secondary)
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
                Button(store.huggingFaceTokenIsSet ? "Replace…" : "Set…") {
                    draft = ""
                    isEditing = true
                }
                if store.huggingFaceTokenIsSet {
                    Button {
                        store.clearHuggingFaceToken()
                    } label: {
                        Image(systemName: "trash")
                    }
                    .help("Remove the stored token")
                }
            }
        }
    }

    private var placeholder: String {
        store.huggingFaceTokenIsSet
            ? "••• token stored in Keychain"
            : "No token set — gated / private HF assets will fail to load"
    }

    private func commit() {
        let trimmed = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        store.setHuggingFaceToken(trimmed)
        draft = ""
        isEditing = false
    }
}
