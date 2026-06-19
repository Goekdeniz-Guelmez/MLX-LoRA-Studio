import Foundation
import Observation
import Security
import SwiftUI

@MainActor
@Observable
final class AppStore {
    var selection: SidebarSection = .train
    var columnVisibility: NavigationSplitViewVisibility = .all
    var training = TrainingConfig()
    var synthetic = SyntheticConfig()
    var hfUpload = HFUploadConfig()
    var trainingRunner = PythonJobRunner()
    var syntheticRunner = PythonJobRunner()
    var hfUploadRunner = PythonJobRunner()
    var packageRunner = PythonJobRunner()
    var runs: [RunRecord] = []
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
    var decorativeAnimationsEnabled = true {
        didSet {
            UserDefaults.standard.set(decorativeAnimationsEnabled, forKey: DefaultsKey.decorativeAnimationsEnabled)
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

    var pythonEnvironments: [PythonEnvironment] = []
    var customPythonPath: String = ""
    var pythonActivationOK: Bool? = nil
    var isScanningPythons: Bool = false
    var isCreatingPythonEnv: Bool = false
    @ObservationIgnored private var activationTask: Task<Void, Never>?

    var cachedModels: [HFCachedAsset] = []
    var cachedDatasets: [HFCachedAsset] = []
    var customModelPaths: [String] = []
    var customDatasetPaths: [String] = []
    var customSystemPrompts: [String] = []
    var isScanningHFCache: Bool = false
    var trainingRunOutputs: [LocalRunOutput] = []
    var syntheticRunOutputs: [LocalRunOutput] = []

    var providerModels: [SyntheticBackend: [String]] = [:]
    var isScanningProviderModels: Set<SyntheticBackend> = []
    var providerModelError: [SyntheticBackend: String] = [:]

    var huggingFaceTokenIsSet: Bool = false

    var syntheticProviderKeyIsSet: [SyntheticBackend: Bool] = [:]

    private enum DefaultsKey {
        static let selectedPythonPath = "selectedPythonPath"
        static let customPythonPath = "customPythonPath"
        static let outputRoot = "outputRoot"
        static let customModelPaths = "customModelPaths"
        static let customDatasetPaths = "customDatasetPaths"
        static let customSystemPrompts = "customSystemPrompts"
        static let decorativeAnimationsEnabled = "decorativeAnimationsEnabled"
        static let completionNotificationsEnabled = "completionNotificationsEnabled"
        static let resourceGuardMemoryPercent = "resourceGuardMemoryPercent"
        static let iogpuWiredLimitMB = "iogpuWiredLimitMB"
        static let onboardingCompleted = "onboardingCompleted"
    }

    private enum KeychainKey {
        static let service = "com.goekdeniz.mlx-lora-studio"
        static let account = "huggingface-token"
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
        if defaults.object(forKey: DefaultsKey.decorativeAnimationsEnabled) != nil {
            decorativeAnimationsEnabled = defaults.bool(forKey: DefaultsKey.decorativeAnimationsEnabled)
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

        huggingFaceTokenIsSet = (readHuggingFaceToken()?.isEmpty == false)

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

    func huggingFaceToken() -> String? {
        readHuggingFaceToken()
    }

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

    func clearHuggingFaceToken() {
        deleteHuggingFaceToken()
        huggingFaceTokenIsSet = false
    }

    // MARK: - Synthetic provider API keys (per-provider Keychain slots)
    func syntheticProviderKey(for backend: SyntheticBackend) -> String? {
        readSyntheticProviderKey(for: backend)
    }

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
    var selectedPythonEnvironment: PythonEnvironment? {
        let canonical = PythonEnvironment.canonicalID(for: pythonExecutable)
        return pythonEnvironments.first { $0.id == canonical }
    }

    func refreshPythonEnvironments() async {
        isScanningPythons = true
        defer { isScanningPythons = false }
        let discovered = await PythonEnvironmentDiscovery.scan()
        pythonEnvironments = discovered.sorted { $0.label.localizedCaseInsensitiveCompare($1.label) == .orderedAscending }
        scheduleActivationCheck()
    }

    func selectPython(_ env: PythonEnvironment) {
        pythonExecutable = env.path
        UserDefaults.standard.set(env.path, forKey: DefaultsKey.selectedPythonPath)
        customPythonPath = ""
        UserDefaults.standard.set("", forKey: DefaultsKey.customPythonPath)
        scheduleActivationCheck()
    }

    func selectCustomPython(_ path: String) {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        customPythonPath = trimmed
        UserDefaults.standard.set(trimmed, forKey: DefaultsKey.customPythonPath)
        guard !trimmed.isEmpty else { return }
        pythonExecutable = trimmed
        UserDefaults.standard.set(trimmed, forKey: DefaultsKey.selectedPythonPath)
        scheduleActivationCheck()
    }

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
                selectCustomPython(newPython.path)
            }
        } catch {
            packageRunner.appendSystemLine("Could not create environment: \(error.localizedDescription)")
        }
    }

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
    func refreshHFCache() async {
        isScanningHFCache = true
        defer { isScanningHFCache = false }
        let scanned = await Task.detached(priority: .userInitiated) {
            HFCacheScanner.scan()
        }.value
        cachedModels = scanned.models
        cachedDatasets = scanned.datasets
    }

    func scrapeProviderModels(backend: SyntheticBackend) async {
        if backend == .mlx { return }
        isScanningProviderModels.insert(backend)
        defer { isScanningProviderModels.remove(backend) }

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
                training.append(LocalRunOutput(name: runURL.lastPathComponent, path: adapters.path))
            }
            let generatedData = runURL.appending(path: "generated-data", directoryHint: .isDirectory)
            if FileManager.default.fileExists(atPath: generatedData.path) {
                synthetic.append(LocalRunOutput(name: runURL.lastPathComponent, path: generatedData.path))
            }
        }
        return (training, synthetic)
    }

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
