import Foundation
import Observation
import Security
import SwiftUI

@MainActor
@Observable
final class AppStore {
    var selection: SidebarSection = .train
    // Visibility of the leading sidebar column. Defaults to `all` so the
    // sidebar is always visible on launch; the user can still toggle it
    // (and the system toggles it on very narrow windows automatically).
    var columnVisibility: NavigationSplitViewVisibility = .all
    var training = TrainingConfig()
    var synthetic = SyntheticConfig()
    var hfUpload = HFUploadConfig()
    var trainingRunner = PythonJobRunner()
    var syntheticRunner = PythonJobRunner()
    var hfUploadRunner = PythonJobRunner()
    var packageRunner = PythonJobRunner()
    var runs: [RunRecord] = []
    /// Runs that the runner has written to the output root in previous
    /// sessions (or earlier in this one). Discovered on launch and
    /// refreshed on demand from the Runs page. See `RunArchive` for
    /// the on-disk layout.
    var persistedRuns: [PersistedRun] = []
    var isRefreshingPersistedRuns: Bool = false
    var isDeletingPersistedRuns: Bool = false
    var pythonExecutable = "/usr/bin/python3"
    var packagePath = ""
    var workingDirectory = ""
    var outputRoot = defaultOutputRoot {
        didSet {
            UserDefaults.standard.set(outputRoot, forKey: DefaultsKey.outputRoot)
        }
    }
    var completionNotificationsEnabled = false {
        didSet {
            UserDefaults.standard.set(completionNotificationsEnabled, forKey: DefaultsKey.completionNotificationsEnabled)
        }
    }
    var isUpdatingCompletionNotifications = false
    var completionNotificationsStatus = ""
    var resourceGuardMemoryPercent = 78 {
        didSet {
            resourceGuardMemoryPercent = Self.clampedResourceGuardMemoryPercent(resourceGuardMemoryPercent)
            UserDefaults.standard.set(resourceGuardMemoryPercent, forKey: DefaultsKey.resourceGuardMemoryPercent)
        }
    }
    var iogpuWiredLimitMB = 21_504 {
        didSet {
            iogpuWiredLimitMB = Self.clampedIOGPUWiredLimitMB(iogpuWiredLimitMB)
            UserDefaults.standard.set(iogpuWiredLimitMB, forKey: DefaultsKey.iogpuWiredLimitMB)
        }
    }
    var iogpuWiredLimitStatus = ""
    var isApplyingIOGPUWiredLimit = false
    var showsOnboarding = false

    // Python interpreter picker state
    var pythonEnvironments: [PythonEnvironment] = []
    var customPythonPath: String = ""
    var pythonActivationOK: Bool? = nil
    var isScanningPythons: Bool = false
    var isCreatingPythonEnv: Bool = false
    @ObservationIgnored private var activationTask: Task<Void, Never>?

    // Hugging Face cache + custom paths for the model/dataset pickers.
    // The cached list is refreshed on demand (Refresh button next to
    // the model/dataset dropdowns) and on the first appearance of the
    // Training view. The custom lists are persisted to UserDefaults so
    // the user's favourite local paths survive across sessions.
    var cachedModels: [HFCachedAsset] = []
    var cachedDatasets: [HFCachedAsset] = []
    var customModelPaths: [String] = []
    var customDatasetPaths: [String] = []
    var customSystemPrompts: [String] = []
    var isScanningHFCache: Bool = false
    var trainingRunOutputs: [LocalRunOutput] = []
    var syntheticRunOutputs: [LocalRunOutput] = []

    // Live-scraped model catalogs for each synthetic-data provider
    // backend (OpenAI / OpenRouter / Ollama / LM Studio / oMLX /
    // Custom). The picker on the synthetic data page reads from
    // here instead of a static "suggested" list, so the dropdown
    // always reflects what the provider actually has right now.
    //
    // `isScanningProviderModels` is the per-backend in-flight flag
    // that the picker binds its refresh button + spinner to, so the
    // user can see when a scrape is in progress and trigger a
    // manual rescrape if a new model shows up mid-session.
    //
    // `providerModelError` is a short human-readable string the
    // picker surfaces under the field when the last scrape failed
    // (e.g. "Provider returned HTTP 401" or "Could not reach
    // Ollama"). It is cleared on the next successful scrape.
    var providerModels: [SyntheticBackend: [String]] = [:]
    var isScanningProviderModels: Set<SyntheticBackend> = []
    var providerModelError: [SyntheticBackend: String] = [:]

    // Hugging Face personal access token. Used to authenticate downloads
    // of gated / private models and datasets (e.g. Llama, your own
    // private repos). Stored in the macOS Keychain so it isn't written
    // to a plain-text plist. The runner copies the value to the
    // HF_TOKEN + HUGGING_FACE_HUB_TOKEN env vars of the spawned Python
    // process; both env-var names are checked by the official
    // `huggingface_hub` library.
    //
    // `huggingFaceTokenIsSet` mirrors "is there a non-empty value in the
    // keychain?" — the store never holds the secret in memory longer
    // than necessary (it reads on demand and forgets), so the UI
    // displays the presence flag, not the value.
    var huggingFaceTokenIsSet: Bool = false

    // Saved API keys for the synthetic-data provider backends
    // (OpenAI / OpenRouter / oMLX / Custom / ...). Each provider has
    // its own Keychain slot so a key for one provider never leaks
    // across — switching from OpenAI to OpenRouter in the picker
    // reveals the new provider's key (or "no key set" if the user
    // never saved one). The keys are NOT written to the system
    // environment or to any shell profile, so they don't collide with
    // `OPENAI_API_KEY` / `OPENROUTER_API_KEY` the user might already
    // have set globally. At run time the runner injects the saved key
    // as a synthetic-specific env var (`SYNTHETIC_OPENAI_API_KEY`)
    // into the Python subprocess; the parent environment is left
    // untouched.
    //
    // The store keeps a per-provider presence flag only — the secret
    // itself is read from the Keychain on demand, then dropped. The
    // UI shows "••• set" or "Paste key…" based on the flag, never the
    // value, mirroring the Hugging Face token field pattern.
    var syntheticProviderKeyIsSet: [SyntheticBackend: Bool] = [:]

    private enum DefaultsKey {
        static let selectedPythonPath = "selectedPythonPath"
        static let customPythonPath = "customPythonPath"
        static let outputRoot = "outputRoot"
        static let customModelPaths = "customModelPaths"
        static let customDatasetPaths = "customDatasetPaths"
        static let customSystemPrompts = "customSystemPrompts"
        static let completionNotificationsEnabled = "completionNotificationsEnabled"
        static let resourceGuardMemoryPercent = "resourceGuardMemoryPercent"
        static let iogpuWiredLimitMB = "iogpuWiredLimitMB"
        static let onboardingCompleted = "onboardingCompleted"
    }

    private enum KeychainKey {
        // Single account for our app's secret; service is the bundle
        // identifier so it can never collide with another app's slot
        // on the same machine.
        static let service = "com.goekdeniz.mlx-lora-studio"
        static let account = "huggingface-token"
        // Per-provider account slot for the synthetic-data API keys.
        // Each provider gets its own keychain entry, so saving an
        // OpenAI key never overwrites a previously-saved OpenRouter
        // key, and the runner reads only the slot that matches the
        // currently selected backend. Account names are namespaced
        // with `synthetic-` so they can never collide with the HF
        // token above.
        static func syntheticAccount(for backend: SyntheticBackend) -> String {
            "synthetic-\(backend.rawValue)-api-key"
        }
    }

    init() {
        let root = ProjectRootResolver.resolve()
        packagePath = root.appending(path: "vendor/mlx-lm-lora").path
        workingDirectory = root.path

        let defaults = UserDefaults.standard
        if let savedOutputRoot = defaults.string(forKey: DefaultsKey.outputRoot),
           !savedOutputRoot.isEmpty {
            outputRoot = savedOutputRoot
        }
        if let saved = defaults.string(forKey: DefaultsKey.selectedPythonPath),
           !saved.isEmpty {
            pythonExecutable = saved
        }
        if let savedCustom = defaults.string(forKey: DefaultsKey.customPythonPath) {
            customPythonPath = savedCustom
        }
        if let savedModels = defaults.array(forKey: DefaultsKey.customModelPaths) as? [String] {
            customModelPaths = savedModels.filter { !$0.isEmpty }
        }
        if let savedDatasets = defaults.array(forKey: DefaultsKey.customDatasetPaths) as? [String] {
            customDatasetPaths = savedDatasets.filter { !$0.isEmpty }
        }
        if let savedSystemPrompts = defaults.array(forKey: DefaultsKey.customSystemPrompts) as? [String] {
            customSystemPrompts = savedSystemPrompts
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        }
        if defaults.object(forKey: DefaultsKey.completionNotificationsEnabled) != nil {
            completionNotificationsEnabled = defaults.bool(forKey: DefaultsKey.completionNotificationsEnabled)
        }
        if defaults.object(forKey: DefaultsKey.resourceGuardMemoryPercent) != nil {
            resourceGuardMemoryPercent = Self.clampedResourceGuardMemoryPercent(
                defaults.integer(forKey: DefaultsKey.resourceGuardMemoryPercent)
            )
        }
        if defaults.object(forKey: DefaultsKey.iogpuWiredLimitMB) != nil {
            iogpuWiredLimitMB = Self.clampedIOGPUWiredLimitMB(
                defaults.integer(forKey: DefaultsKey.iogpuWiredLimitMB)
            )
        }
        showsOnboarding = !defaults.bool(forKey: DefaultsKey.onboardingCompleted)

        // The presence flag is a single bit we keep in memory; the
        // secret itself is read from the Keychain on demand by
        // `huggingFaceToken()`. The Keychain is queried once on launch
        // so the UI can show "HF key set" / "HF key missing" without
        // blocking on every render.
        huggingFaceTokenIsSet = (readHuggingFaceToken()?.isEmpty == false)

        // Same trick for the synthetic provider keys: query each
        // Keychain slot once at launch so the synthetic data page can
        // show the right "key set" / "no key" indicator for the
        // currently-selected backend without blocking on every render.
        // The actual secret value is only read on demand from
        // `syntheticProviderKey(for:)` right before a run.
        for backend in SyntheticBackend.allCases {
            syntheticProviderKeyIsSet[backend] =
                (readSyntheticProviderKey(for: backend)?.isEmpty == false)
        }

        Task { await refreshPythonEnvironments() }
        Task { await refreshHFCache() }
        Task { await refreshLocalRunOutputs() }
        Task { await refreshPersistedRuns() }
    }

    func completeOnboarding() {
        showsOnboarding = false
        UserDefaults.standard.set(true, forKey: DefaultsKey.onboardingCompleted)
    }

    func replayOnboarding() {
        showsOnboarding = true
    }

    // MARK: - Hugging Face token (Keychain-backed)

    private static var defaultOutputRoot: String {
        URL(fileURLWithPath: NSHomeDirectory())
            .appending(path: ".mlxlorastudio", directoryHint: .isDirectory)
            .appending(path: "runs", directoryHint: .isDirectory)
            .path
    }

    private static func clampedResourceGuardMemoryPercent(_ value: Int) -> Int {
        min(max(value, 10), 98)
    }

    private static func clampedIOGPUWiredLimitMB(_ value: Int) -> Int {
        min(max(value, 1_024), 1_048_576)
    }

    func applyIOGPUWiredLimit() async {
        isApplyingIOGPUWiredLimit = true
        iogpuWiredLimitStatus = "Waiting for administrator approval..."
        defer { isApplyingIOGPUWiredLimit = false }

        do {
            let result = try await IOGPUWiredLimitService.apply(limitMB: iogpuWiredLimitMB)
            iogpuWiredLimitStatus = result.isEmpty
                ? "Applied iogpu.wired_limit_mb=\(iogpuWiredLimitMB)."
                : result
        } catch {
            iogpuWiredLimitStatus = "Could not apply iogpu.wired_limit_mb: \(error.localizedDescription)"
        }
    }

    /// Reads the current HF token from the Keychain. Returns nil if no
    /// token has been stored. Callers should treat the return value as
    /// ephemeral — do not retain it in a stored property. `runner`
    /// reads it right before launching a process and then drops it.
    func huggingFaceToken() -> String? {
        readHuggingFaceToken()
    }

    /// Stores (or replaces) the HF token in the Keychain. Passing an
    /// empty / whitespace-only string clears the entry instead. Updates
    /// `huggingFaceTokenIsSet` so observers re-render the "set" / "not
    /// set" indicator without needing to know about the Keychain.
    func setHuggingFaceToken(_ token: String) {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            deleteHuggingFaceToken()
            huggingFaceTokenIsSet = false
        } else {
            writeHuggingFaceToken(trimmed)
            huggingFaceTokenIsSet = true
        }
    }

    /// Removes the stored HF token entirely. Called by the "Clear"
    /// button next to the field. The change is observable through
    /// `huggingFaceTokenIsSet`, which any open view reacts to.
    func clearHuggingFaceToken() {
        deleteHuggingFaceToken()
        huggingFaceTokenIsSet = false
    }

    // MARK: - Synthetic provider API keys (per-provider Keychain slots)

    /// Reads the saved API key for a synthetic-data provider, or
    /// returns nil if none is set. The value is read on demand from
    /// the Keychain and the caller is expected to drop it after use
    /// — the store does not retain the secret in memory. Only the
    /// runner needs this, and only at the moment it injects the key
    /// into the spawned Python subprocess.
    func syntheticProviderKey(for backend: SyntheticBackend) -> String? {
        readSyntheticProviderKey(for: backend)
    }

    /// Stores (or replaces) the API key for a synthetic-data provider
    /// in the Keychain. An empty / whitespace-only string clears the
    /// slot. Updates the per-provider `syntheticProviderKeyIsSet`
    /// flag so the synthetic data page can show the right "key set"
    /// / "no key" indicator without having to know about the
    /// Keychain.
    ///
    /// Important: this never writes to the system environment or to
    /// any shell profile. Each provider has its own Keychain slot,
    /// so saving an OpenAI key cannot overwrite a previously-saved
    /// OpenRouter key. The runner reads only the slot that matches
    /// the currently selected backend.
    func setSyntheticProviderKey(_ key: String, for backend: SyntheticBackend) {
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            deleteSyntheticProviderKey(for: backend)
            syntheticProviderKeyIsSet[backend] = false
        } else {
            writeSyntheticProviderKey(trimmed, for: backend)
            syntheticProviderKeyIsSet[backend] = true
        }
    }

    /// Removes the stored API key for a synthetic-data provider. The
    /// change is observable through `syntheticProviderKeyIsSet[backend]`,
    /// so the synthetic data page can re-render the "no key" state
    /// the moment the trash button is pressed.
    func clearSyntheticProviderKey(for backend: SyntheticBackend) {
        deleteSyntheticProviderKey(for: backend)
        syntheticProviderKeyIsSet[backend] = false
    }

    private func readSyntheticProviderKey(for backend: SyntheticBackend) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: KeychainKey.service,
            kSecAttrAccount as String: KeychainKey.syntheticAccount(for: backend),
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var item: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess, let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private func writeSyntheticProviderKey(_ key: String, for backend: SyntheticBackend) {
        let data = Data(key.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: KeychainKey.service,
            kSecAttrAccount as String: KeychainKey.syntheticAccount(for: backend)
        ]
        let updateAttrs: [String: Any] = [
            kSecValueData as String: data
        ]
        let updateStatus = SecItemUpdate(query as CFDictionary, updateAttrs as CFDictionary)
        if updateStatus == errSecItemNotFound {
            var addQuery = query
            addQuery[kSecValueData as String] = data
            // Mirror the HF-token policy: pin to the user's login
            // keychain (no iCloud sync) and only readable while
            // logged in.
            addQuery[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlocked
            SecItemAdd(addQuery as CFDictionary, nil)
        }
    }

    private func deleteSyntheticProviderKey(for backend: SyntheticBackend) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: KeychainKey.service,
            kSecAttrAccount as String: KeychainKey.syntheticAccount(for: backend)
        ]
        SecItemDelete(query as CFDictionary)
    }

    private func readHuggingFaceToken() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: KeychainKey.service,
            kSecAttrAccount as String: KeychainKey.account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var item: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess, let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private func writeHuggingFaceToken(_ token: String) {
        let data = Data(token.utf8)
        // Try update first; if no item exists, fall back to add. This
        // handles the "replace" case (user types a new token) and the
        // first-time "store" case with a single code path.
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: KeychainKey.service,
            kSecAttrAccount as String: KeychainKey.account
        ]
        let updateAttrs: [String: Any] = [
            kSecValueData as String: data
        ]
        let updateStatus = SecItemUpdate(query as CFDictionary, updateAttrs as CFDictionary)
        if updateStatus == errSecItemNotFound {
            var addQuery = query
            addQuery[kSecValueData as String] = data
            // Available on macOS — pin to the user's login keychain so
            // the value persists across launches but is not synced to
            // iCloud. `kSecAttrAccessibleWhenUnlocked` means the
            // secret is only readable while the user is logged in.
            addQuery[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlocked
            SecItemAdd(addQuery as CFDictionary, nil)
        }
    }

    private func deleteHuggingFaceToken() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: KeychainKey.service,
            kSecAttrAccount as String: KeychainKey.account
        ]
        SecItemDelete(query as CFDictionary)
    }

    func startSelectedJob() async {
        switch selection {
        case .train:
            await startTraining()
        case .synthetic:
            await startSynthetic()
        case .upload:
            await startHFUpload()
        case .metrics, .guide, .runs, .about:
            break
        }
    }

    func startTraining() async {
        let runID = UUID()
        do {
            let command = try await trainingRunner.startTraining(
                config: training,
                pythonExecutable: pythonExecutable,
                packagePath: packagePath,
                workingDirectory: workingDirectory,
                outputRoot: outputRoot,
                resourceGuardMemoryPercent: resourceGuardMemoryPercent,
                huggingFaceToken: huggingFaceToken(),
                onCompletion: jobCompletionHandler(
                    runID: runID,
                    successTitle: "\(training.trainMode.title) training finished",
                    failureTitle: "\(training.trainMode.title) training stopped",
                    successBody: "The training run has completed.",
                    failureBody: "The training run exited before completing successfully."
                )
            )
            runs.insert(
                RunRecord(id: runID, title: "\(training.trainMode.title) Training", command: command, startedAt: .now, status: "Running"),
                at: 0
            )
        } catch {
            trainingRunner.appendSystemLine("Could not start training: \(error.localizedDescription)")
        }
    }

    func resumeTraining(from run: PersistedRun) async {
        guard var config = run.spec,
              let candidate = RunArchive.resumeCandidate(for: run) else {
            trainingRunner.appendSystemLine("Could not resume run: no saved adapter checkpoint was found.")
            return
        }

        config.resumeAdapterFile = candidate.adapterFile.path
        config.runFolderName = resumeRunFolderName(for: run, step: candidate.step)
        if config.epochs == 0, let step = candidate.step, config.iters > step {
            config.iters -= step
        }
        training = config
        selection = .train
        trainingRunner.appendSystemLine("Preparing resume from \(candidate.adapterFile.path)")
        await startTraining()
    }

    func startSynthetic() async {
        let runID = UUID()
        do {
            // Pull the per-provider saved API key from the Keychain
            // right before launching, so the secret lives in memory
            // for the minimum possible time. The runner drops it as
            // soon as it has injected it into the spawned Python
            // subprocess's env dict. A non-empty value the user typed
            // into the form (`synthetic.apiKey`) overrides the saved
            // key — see `PythonJobRunner.startSynthetic`.
            let savedKey = syntheticProviderKey(for: synthetic.backend)
            let command = try await syntheticRunner.startSynthetic(
                config: synthetic,
                pythonExecutable: pythonExecutable,
                packagePath: packagePath,
                workingDirectory: workingDirectory,
                outputRoot: outputRoot,
                huggingFaceToken: huggingFaceToken(),
                savedSyntheticAPIKey: savedKey,
                onCompletion: jobCompletionHandler(
                    runID: runID,
                    successTitle: "\(synthetic.kind.title) data generation finished",
                    failureTitle: "\(synthetic.kind.title) data generation stopped",
                    successBody: "The synthetic data job has completed.",
                    failureBody: "The synthetic data job exited before completing successfully."
                )
            )
            runs.insert(
                RunRecord(id: runID, title: "\(synthetic.kind.title) Synthetic Data", command: command, startedAt: .now, status: "Running"),
                at: 0
            )
        } catch {
            syntheticRunner.appendSystemLine("Could not start synthetic job: \(error.localizedDescription)")
        }
    }

    func startHFUpload(target: HFUploadTarget = .all) async {
        let runID = UUID()
        normalizeHFUploadKindForSelectedModel()
        do {
            let command = try await hfUploadRunner.startHFUpload(
                config: hfUpload,
                target: target,
                pythonExecutable: pythonExecutable,
                packagePath: packagePath,
                workingDirectory: workingDirectory,
                outputRoot: outputRoot,
                huggingFaceToken: huggingFaceToken(),
                onCompletion: jobCompletionHandler(
                    runID: runID,
                    successTitle: "Hugging Face upload finished",
                    failureTitle: "Hugging Face upload stopped",
                    successBody: "The upload job has completed.",
                    failureBody: "The upload job exited before completing successfully."
                )
            )
            runs.insert(
                RunRecord(id: runID, title: "\(target.title) HF Upload", command: command, startedAt: .now, status: "Running"),
                at: 0
            )
        } catch {
            hfUploadRunner.appendSystemLine("Could not start HF upload: \(error.localizedDescription)")
        }
    }

    func updatePackages() async {
        let runID = UUID()
        do {
            let command = try await packageRunner.startPackageUpdate(
                pythonExecutable: pythonExecutable,
                packagePath: packagePath,
                workingDirectory: workingDirectory,
                onCompletion: jobCompletionHandler(runID: runID)
            )
            runs.insert(
                RunRecord(id: runID, title: "Package Update", command: command, startedAt: .now, status: "Running"),
                at: 0
            )
        } catch {
            packageRunner.appendSystemLine("Could not start package update: \(error.localizedDescription)")
        }
    }

    func reinstallPackages() async {
        let runID = UUID()
        do {
            let command = try await packageRunner.startPackageUpdate(
                pythonExecutable: pythonExecutable,
                packagePath: packagePath,
                workingDirectory: workingDirectory,
                forceReinstall: true,
                onCompletion: jobCompletionHandler(runID: runID)
            )
            runs.insert(
                RunRecord(id: runID, title: "Package Reinstall", command: command, startedAt: .now, status: "Running"),
                at: 0
            )
        } catch {
            packageRunner.appendSystemLine("Could not start package reinstall: \(error.localizedDescription)")
        }
    }

    func stopJob() {
        selectedRunner?.stop()
        if !runs.isEmpty {
            runs[0].status = "Stopped"
            runs[0].endedAt = .now
        }
    }

    func togglePlayback() async {
        if let runner = selectedRunner, runner.isRunning {
            if runner.isPaused {
                runner.resume()
                if !runs.isEmpty { runs[0].status = "Running" }
            } else {
                runner.pause()
                if !runs.isEmpty { runs[0].status = "Paused" }
            }
        } else {
            await startSelectedJob()
        }
    }

    /// Toggles the **training** runner specifically. Used by the
    /// upper-right toolbar and the ⌘↩ / ⌘. menu-bar shortcuts, which
    /// are now decoupled from the Synthetic and Upload runners —
    /// those pages have their own Start / Cancel pills at the top of
    /// the page and don't want a global control stealing their
    /// cancel action.
    func toggleTrainingPlayback() async {
        if trainingRunner.isRunning {
            if trainingRunner.isPaused {
                trainingRunner.resume()
                if !runs.isEmpty { runs[0].status = "Running" }
            } else {
                trainingRunner.pause()
                if !runs.isEmpty { runs[0].status = "Paused" }
            }
        } else {
            await startTraining()
        }
    }

    var selectedRunner: PythonJobRunner? {
        switch selection {
        case .train:
            trainingRunner
        case .synthetic:
            syntheticRunner
        case .upload:
            hfUploadRunner
        case .metrics:
            trainingRunner
        case .guide, .runs, .about:
            nil
        }
    }

    var allJobRunners: [PythonJobRunner] {
        [trainingRunner, syntheticRunner, hfUploadRunner, packageRunner]
    }

    var anyJobRunning: Bool {
        allJobRunners.contains { $0.isRunning }
    }

    var canStartSelectedJob: Bool {
        switch selection {
        case .train, .synthetic, .upload:
            true
        case .metrics, .guide, .runs, .about:
            false
        }
    }

    func setCompletionNotificationsEnabled(_ enabled: Bool) async {
        guard enabled != completionNotificationsEnabled else { return }

        if !enabled {
            completionNotificationsEnabled = false
            completionNotificationsStatus = "Job completion notifications are off."
            return
        }

        guard JobCompletionNotifier.isAvailable else {
            completionNotificationsEnabled = false
            completionNotificationsStatus = "Notifications are only available when MLX LoRA Studio is running from the built .app bundle."
            return
        }

        isUpdatingCompletionNotifications = true
        completionNotificationsStatus = "Requesting macOS notification permission..."
        let granted = await JobCompletionNotifier.requestAuthorizationIfNeeded()
        isUpdatingCompletionNotifications = false
        completionNotificationsEnabled = granted
        completionNotificationsStatus = granted
            ? "Job completion notifications are on."
            : "macOS did not grant notification permission. Enable MLX LoRA Studio in System Settings > Notifications, then try again."
    }

    private func jobCompletionHandler(
        runID: UUID,
        successTitle: String? = nil,
        failureTitle: String? = nil,
        successBody: String? = nil,
        failureBody: String? = nil
    ) -> @MainActor (Int32) -> Void {
        { [weak self] status in
            guard let self else { return }
            markRun(runID, status: status == 0 ? "Finished" : "Failed")
            guard completionNotificationsEnabled,
                  let successTitle,
                  let failureTitle,
                  let successBody,
                  let failureBody else { return }
            let succeeded = status == 0
            Task {
                await JobCompletionNotifier.send(
                    title: succeeded ? successTitle : failureTitle,
                    body: succeeded ? successBody : "\(failureBody) Exit status: \(status)."
                )
            }
        }
    }

    private func markRun(_ id: UUID, status: String) {
        guard let index = runs.firstIndex(where: { $0.id == id }) else { return }
        runs[index].status = status
        runs[index].endedAt = .now
    }

    // MARK: - Python environment picker

    /// Currently-selected env, or `nil` if the user is on a custom path that
    /// wasn't in the discovered set. The picker uses this to show a checkmark.
    var selectedPythonEnvironment: PythonEnvironment? {
        let canonical = PythonEnvironment.canonicalID(for: pythonExecutable)
        return pythonEnvironments.first { $0.id == canonical }
    }

    /// Re-scans the system for Python interpreters. Safe to call repeatedly;
    /// the picker disables its Refresh button while a scan is in flight.
    func refreshPythonEnvironments() async {
        isScanningPythons = true
        defer { isScanningPythons = false }
        let discovered = await PythonEnvironmentDiscovery.scan()
        pythonEnvironments = discovered.sorted { $0.label.localizedCaseInsensitiveCompare($1.label) == .orderedAscending }
        scheduleActivationCheck()
    }

    /// Picks one of the discovered environments. Persists the choice and
    /// re-runs the activation probe.
    func selectPython(_ env: PythonEnvironment) {
        pythonExecutable = env.path
        UserDefaults.standard.set(env.path, forKey: DefaultsKey.selectedPythonPath)
        customPythonPath = ""
        UserDefaults.standard.set("", forKey: DefaultsKey.customPythonPath)
        scheduleActivationCheck()
    }

    /// Switches to a custom path typed by the user. The picker always shows
    /// the custom field when this is non-empty.
    func selectCustomPython(_ path: String) {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        customPythonPath = trimmed
        UserDefaults.standard.set(trimmed, forKey: DefaultsKey.customPythonPath)
        guard !trimmed.isEmpty else { return }
        pythonExecutable = trimmed
        UserDefaults.standard.set(trimmed, forKey: DefaultsKey.selectedPythonPath)
        scheduleActivationCheck()
    }

    /// Creates a new env, refreshes the picker, and selects the new env.
    func createPythonEnv(name: String, kind: PythonEnvironment.Kind, base: PythonEnvironment) async {
        isCreatingPythonEnv = true
        defer { isCreatingPythonEnv = false }
        let log: @MainActor @Sendable (String) -> Void = { [weak self] line in
            self?.packageRunner.appendSystemLine(line)
        }
        do {
            let newPython: URL
            switch kind {
            case .conda:
                newPython = try await PythonEnvironmentProvisioner.createCondaEnv(
                    name: name,
                    basePython: base.path,
                    log: log
                )
            default:
                newPython = try await PythonEnvironmentProvisioner.createVenv(
                    name: name,
                    basePython: base.path,
                    parentDir: PythonEnvironmentProvisioner.defaultVenvParent(),
                    log: log
                )
            }
            await refreshPythonEnvironments()
            if let env = pythonEnvironments.first(where: { $0.id == PythonEnvironment.canonicalID(for: newPython.path) }) {
                selectPython(env)
            } else {
                // Fall back to a custom path if discovery didn't catch the new env.
                selectCustomPython(newPython.path)
            }
        } catch {
            packageRunner.appendSystemLine("Could not create environment: \(error.localizedDescription)")
        }
    }

    /// Runs `python -c "import mlx_lm_lora"` against the active interpreter on
    /// a 600 ms debounce so a fast picker scrub doesn't spawn a dozen probes.
    private func scheduleActivationCheck() {
        activationTask?.cancel()
        let executable = pythonExecutable
        pythonActivationOK = nil
        activationTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 600_000_000)
            if Task.isCancelled { return }
            let ok = await PythonEnvironmentDiscovery.runActivationCheck(executable: executable)
            if Task.isCancelled { return }
            self?.pythonActivationOK = ok
        }
    }

    // MARK: - Hugging Face cache picker

    /// Re-scans the local HF cache for downloaded models and datasets.
    /// Safe to call repeatedly; the picker disables its Refresh button
    /// while a scan is in flight. The scan itself is fast (one
    /// `contentsOfDirectory` per asset) so we just hop off the main
    /// actor briefly.
    func refreshHFCache() async {
        isScanningHFCache = true
        defer { isScanningHFCache = false }
        let scanned = await Task.detached(priority: .userInitiated) {
            HFCacheScanner.scan()
        }.value
        cachedModels = scanned.models
        cachedDatasets = scanned.datasets
    }

    /// Re-scrapes the live model catalog for one synthetic-data
    /// provider backend. Called automatically by the picker when it
    /// first appears and again whenever the user switches to a
    /// different backend; the picker also has a manual refresh
    /// button that re-invokes this for the rare case where a new
    /// model shows up mid-session.
    ///
    /// MLX is a no-op (it has no remote catalog — the user types a
    /// local path into the `model` field). The base URL is read
    /// from `synthetic.baseURL`, falling back to the provider's
    /// default if the user cleared it. The auth key is read from
    /// the Keychain slot for this backend, so cloud providers
    /// (OpenAI / OpenRouter / oMLX) authenticate automatically and
    /// local servers (Ollama / LM Studio) do not.
    ///
    /// Failures are recorded into `providerModelError[backend]`
    /// (a short human-readable message) and the previous catalog
    /// is left intact — the user keeps whatever they had so a
    /// transient blip doesn't wipe the dropdown.
    func scrapeProviderModels(backend: SyntheticBackend) async {
        if backend == .mlx { return }
        isScanningProviderModels.insert(backend)
        defer { isScanningProviderModels.remove(backend) }

        // Pull the auth key from the Keychain right before the
        // request, so the secret is in memory for the minimum
        // possible time. We don't retain it in any stored
        // property — the scraper reads it, sends it, and forgets.
        let key = syntheticProviderKey(for: backend)
        let baseURL = synthetic.baseURL

        do {
            let ids = try await ProviderModelCatalog.scrape(
                backend: backend,
                baseURL: baseURL,
                apiKey: key
            )
            providerModels[backend] = ids
            applySyntheticModelFallback(for: backend, models: ids)
            providerModelError[backend] = nil
        } catch {
            providerModelError[backend] = error.localizedDescription
        }
    }

    func refreshLocalRunOutputs() async {
        let root = outputRoot
        let outputs = await Task.detached(priority: .userInitiated) {
            Self.discoverLocalRunOutputs(outputRoot: root)
        }.value
        trainingRunOutputs = outputs.training
        syntheticRunOutputs = outputs.synthetic
        normalizeHFUploadKindForSelectedModel()
    }

    var hfUploadAdaptersUnavailable: Bool {
        let path = hfUpload.localModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        return !path.isEmpty && !Self.hasAdapterWeights(forUploadPath: path, outputs: trainingRunOutputs)
    }

    func normalizeHFUploadKindForSelectedModel() {
        if hfUploadAdaptersUnavailable, hfUpload.modelUploadKind == .adaptersOnly {
            hfUpload.modelUploadKind = .mergedWeights
        }
    }

    func loadSyntheticConfig(fromGeneratedDataPath path: String) {
        let cleanPath = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanPath.isEmpty else { return }
        let generatedDataURL = URL(fileURLWithPath: NSString(string: cleanPath).expandingTildeInPath, isDirectory: true)
        let specURL = generatedDataURL
            .deletingLastPathComponent()
            .appending(path: "synthetic_spec.json")
        guard var decoded = SyntheticConfig.decoded(from: specURL) else {
            synthetic.resumeOutputDir = cleanPath
            syntheticRunner.appendSystemLine("Could not load previous synthetic settings: \(specURL.path)")
            return
        }
        decoded.resumeOutputDir = cleanPath
        decoded.runFolderName = ""
        decoded.apiKey = synthetic.apiKey
        let specModelWasEmpty = (decoded.kind == .sft ? decoded.model : decoded.teacherModel)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty
        applySyntheticModelFallback(to: &decoded, models: providerModels[decoded.backend] ?? [])
        synthetic = decoded
        let restoredModel = decoded.kind == .sft ? decoded.model : decoded.teacherModel
        let modelNote = specModelWasEmpty ? "fallback model" : "model"
        syntheticRunner.appendSystemLine(
            "Loaded synthetic settings from \(specURL.path) — backend: \(decoded.backend.title), \(modelNote): \(restoredModel)"
        )
    }

    private func applySyntheticModelFallback(for backend: SyntheticBackend, models: [String]) {
        guard synthetic.backend == backend else { return }
        applySyntheticModelFallback(to: &synthetic, models: models)
    }

    private func applySyntheticModelFallback(to config: inout SyntheticConfig, models: [String]) {
        guard config.backend != .mlx,
              let fallback = models.first?.trimmingCharacters(in: .whitespacesAndNewlines),
              !fallback.isEmpty else { return }
        switch config.kind {
        case .sft:
            if config.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                config.model = fallback
            }
        case .dpo:
            if config.teacherModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                config.teacherModel = fallback
            }
        }
    }

    /// Re-scans the output root for previous runs and replaces
    /// `persistedRuns` with the result. The disk walk is hoisted off
    /// the main actor so a large runs folder doesn't stall the UI; the
    /// assignment is a single `replace-all` so the Runs page sees a
    /// consistent snapshot, not a half-rendered list.
    ///
    /// Safe to call repeatedly — the Runs page wires a Refresh button
    /// to it, and `init()` calls it once on launch.
    func refreshPersistedRuns() async {
        isRefreshingPersistedRuns = true
        defer { isRefreshingPersistedRuns = false }
        let root = outputRoot
        let runs = await Task.detached(priority: .userInitiated) {
            RunArchive.discoverPersistedRuns(outputRoot: root)
        }.value
        persistedRuns = runs
    }

    func deletePersistedRun(_ run: PersistedRun) async throws {
        let root = outputRoot
        try await Task.detached(priority: .userInitiated) {
            try RunArchive.deletePersistedRun(run, outputRoot: root)
        }.value
        persistedRuns.removeAll { $0.id == run.id }
        await refreshLocalRunOutputs()
    }

    func deletePersistedRuns(_ runs: [PersistedRun]) async throws {
        guard !runs.isEmpty else { return }
        isDeletingPersistedRuns = true
        defer { isDeletingPersistedRuns = false }

        let root = outputRoot
        let deletedIDs = try await Task.detached(priority: .userInitiated) {
            var deletedIDs: [String] = []
            for run in runs {
                try RunArchive.deletePersistedRun(run, outputRoot: root)
                deletedIDs.append(run.id)
            }
            return deletedIDs
        }.value

        persistedRuns.removeAll { deletedIDs.contains($0.id) }
        await refreshLocalRunOutputs()
    }

    func resumeCandidate(for run: PersistedRun) -> TrainingResumeCandidate? {
        RunArchive.resumeCandidate(for: run)
    }

    func continuationCandidate(for run: PersistedRun) -> TrainingResumeCandidate? {
        RunArchive.continuationCandidate(for: run)
    }

    private func resumeRunFolderName(for run: PersistedRun, step: Int?) -> String {
        let stepSuffix = step.map { "-from-\($0)" } ?? ""
        return RunFolderNamer.sanitize("\(run.id)-resume\(stepSuffix)")
    }

    nonisolated private static func discoverLocalRunOutputs(outputRoot: String) -> (training: [LocalRunOutput], synthetic: [LocalRunOutput]) {
        let expandedRoot = NSString(string: outputRoot).expandingTildeInPath
        let rootURL = URL(fileURLWithPath: expandedRoot, isDirectory: true)
        guard let children = try? FileManager.default.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: [.contentModificationDateKey, .isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return ([], [])
        }

        let sorted = children.sorted { lhs, rhs in
            let leftDate = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let rightDate = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return leftDate > rightDate
        }

        var training: [LocalRunOutput] = []
        var synthetic: [LocalRunOutput] = []
        for runURL in sorted {
            guard ((try? runURL.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false) else { continue }
            let adapters = runURL.appending(path: "adapters", directoryHint: .isDirectory)
            if FileManager.default.fileExists(atPath: adapters.path) {
                training.append(
                    LocalRunOutput(
                        name: runURL.lastPathComponent,
                        path: adapters.path,
                        hasAdapters: hasAdapterWeights(in: adapters)
                    )
                )
            }
            let generatedData = runURL.appending(path: "generated-data", directoryHint: .isDirectory)
            if FileManager.default.fileExists(atPath: generatedData.path) {
                synthetic.append(LocalRunOutput(name: runURL.lastPathComponent, path: generatedData.path))
            }
        }
        return (training, synthetic)
    }

    nonisolated static func hasAdapterWeights(in folderURL: URL) -> Bool {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: folderURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return false
        }

        return files.contains { url in
            guard ((try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) ?? false) else {
                return false
            }
            let filename = url.lastPathComponent
            return filename == "adapters.safetensors" || filename.hasSuffix("_adapters.safetensors")
        }
    }

    nonisolated private static func hasAdapterWeights(
        forUploadPath path: String,
        outputs: [LocalRunOutput]
    ) -> Bool {
        if let output = outputs.first(where: { $0.path == path }) {
            return output.hasAdapters
        }
        let expanded = NSString(string: path).expandingTildeInPath
        return hasAdapterWeights(in: URL(fileURLWithPath: expanded, isDirectory: true))
    }

    /// Adds a user-typed HF repo id to the custom list (used by the
    /// "Add HF repo…" sheet on the model/dataset dropdowns). Trims
    /// whitespace, ignores empty strings, and de-duplicates against
    /// the cache and the existing custom list.
    func addCustomHFAsset(_ id: String, kind: HFCachedAsset.Kind) {
        let trimmed = id.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        // Accept both `owner/name` and `owner--name` to be forgiving.
        let normalised = trimmed.contains("/")
            ? trimmed
            : trimmed.replacingOccurrences(of: "--", with: "/")
        guard normalised.contains("/") else { return }
        switch kind {
        case .model:
            guard !cachedModels.contains(where: { $0.hfID == normalised }),
                  !customModelPaths.contains(normalised) else { return }
            customModelPaths.append(normalised)
            UserDefaults.standard.set(customModelPaths, forKey: DefaultsKey.customModelPaths)
        case .dataset:
            guard !cachedDatasets.contains(where: { $0.hfID == normalised }),
                  !customDatasetPaths.contains(normalised) else { return }
            customDatasetPaths.append(normalised)
            UserDefaults.standard.set(customDatasetPaths, forKey: DefaultsKey.customDatasetPaths)
        }
    }

    /// Adds a local file system path the user picked with NSOpenPanel
    /// (or typed into the "Add local path…" sheet). Stores both the
    /// absolute path and the original (un-trimmed) so the user can see
    /// what they added.
    func addCustomLocalPath(_ path: String, kind: HFCachedAsset.Kind) {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        switch kind {
        case .model:
            guard !customModelPaths.contains(trimmed) else { return }
            customModelPaths.append(trimmed)
            UserDefaults.standard.set(customModelPaths, forKey: DefaultsKey.customModelPaths)
        case .dataset:
            guard !customDatasetPaths.contains(trimmed) else { return }
            customDatasetPaths.append(trimmed)
            UserDefaults.standard.set(customDatasetPaths, forKey: DefaultsKey.customDatasetPaths)
        }
    }

    /// Drops a custom path the user no longer wants. Called by the
    /// × button on each custom-path chip in the picker popover.
    func removeCustomPath(_ path: String, kind: HFCachedAsset.Kind) {
        switch kind {
        case .model:
            customModelPaths.removeAll { $0 == path }
            UserDefaults.standard.set(customModelPaths, forKey: DefaultsKey.customModelPaths)
        case .dataset:
            customDatasetPaths.removeAll { $0 == path }
            UserDefaults.standard.set(customDatasetPaths, forKey: DefaultsKey.customDatasetPaths)
        }
    }

    /// Saves a reusable system prompt for the Synthetic Data page.
    /// Prompts are stored as plain text in UserDefaults so the dropdown
    /// survives app launches and stays lightweight.
    func addCustomSystemPrompt(_ prompt: String) {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              !customSystemPrompts.contains(trimmed) else { return }
        customSystemPrompts.append(trimmed)
        UserDefaults.standard.set(customSystemPrompts, forKey: DefaultsKey.customSystemPrompts)
    }

    func removeCustomSystemPrompt(_ prompt: String) {
        customSystemPrompts.removeAll { $0 == prompt }
        UserDefaults.standard.set(customSystemPrompts, forKey: DefaultsKey.customSystemPrompts)
    }
}

enum ProjectRootResolver {
    static func resolve() -> URL {
        let fileManager = FileManager.default
        let candidates = [
            Bundle.main.resourceURL?.appending(path: "StudioSupport", directoryHint: .isDirectory),
            Bundle.main.resourceURL,
            Bundle.main.bundleURL,
            Bundle.main.bundleURL.deletingLastPathComponent(),
            Bundle.main.bundleURL.deletingLastPathComponent().deletingLastPathComponent(),
            URL(fileURLWithPath: fileManager.currentDirectoryPath)
        ].compactMap { $0 }

        for candidate in candidates {
            if let root = nearestProjectRoot(from: candidate, fileManager: fileManager) {
                return root
            }
        }

        return URL(fileURLWithPath: NSHomeDirectory()).appending(path: "MLXLoRAStudio", directoryHint: .isDirectory)
    }

    private static func nearestProjectRoot(from start: URL, fileManager: FileManager) -> URL? {
        var url = start
        for _ in 0..<8 {
            let packageURL = url.appending(path: "vendor/mlx-lm-lora", directoryHint: .isDirectory)
            if fileManager.fileExists(atPath: packageURL.path) {
                return url
            }
            let parent = url.deletingLastPathComponent()
            if parent.path == url.path { break }
            url = parent
        }
        return nil
    }
}
