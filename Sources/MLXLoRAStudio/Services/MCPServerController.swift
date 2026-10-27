import Foundation
import Observation
import Security

@MainActor
@Observable
final class MCPServerController {
    private enum DefaultsKey {
        static let bindAddress = "mcpServerBindAddress"
        static let clientAddress = "mcpServerClientAddress"
        static let port = "mcpServerPort"
        static let pythonExecutable = "mcpServerPythonExecutable"
        static let environmentDirectory = "mcpServerEnvironmentDirectory"
        static let runsDirectory = "mcpServerRunsDirectory"
        static let allowedHosts = "mcpServerAllowedHosts"
        static let bearerAuthentication = "mcpServerBearerAuthentication"
    }

    private enum KeychainKey {
        static let service = "com.goekdeniz.mlx-lora-studio"
        static let account = "mcp-server-bearer-token"
    }

    var bindAddress = "127.0.0.1" {
        didSet { UserDefaults.standard.set(bindAddress, forKey: DefaultsKey.bindAddress) }
    }
    var clientAddress = "127.0.0.1" {
        didSet { UserDefaults.standard.set(clientAddress, forKey: DefaultsKey.clientAddress) }
    }
    var port = 8765 {
        didSet {
            port = min(max(port, 1), 65_535)
            UserDefaults.standard.set(port, forKey: DefaultsKey.port)
        }
    }
    var pythonExecutable = "/usr/bin/python3" {
        didSet { UserDefaults.standard.set(pythonExecutable, forKey: DefaultsKey.pythonExecutable) }
    }
    var environmentDirectory = "~/.mlx-lm-lora/studio-mcp-env" {
        didSet { UserDefaults.standard.set(environmentDirectory, forKey: DefaultsKey.environmentDirectory) }
    }
    var runsDirectory = "~/.mlx-lm-lora/runs" {
        didSet { UserDefaults.standard.set(runsDirectory, forKey: DefaultsKey.runsDirectory) }
    }
    var allowedHosts = "" {
        didSet { UserDefaults.standard.set(allowedHosts, forKey: DefaultsKey.allowedHosts) }
    }
    var bearerAuthenticationEnabled = false {
        didSet { UserDefaults.standard.set(bearerAuthenticationEnabled, forKey: DefaultsKey.bearerAuthentication) }
    }

    private(set) var bearerTokenIsSet = false
    private(set) var isInstalling = false
    private(set) var isRunning = false
    private(set) var status = "MCP server is stopped."
    private(set) var serverOutput: [String] = []
    private(set) var clientStatus: [String: String] = [:]
    private(set) var skillStatus: [String: String] = [:]

    @ObservationIgnored private var serverProcess: Process?
    @ObservationIgnored private var serverOutputPipe: Pipe?

    init() {
        let defaults = UserDefaults.standard
        if let value = defaults.string(forKey: DefaultsKey.bindAddress), !value.isEmpty { bindAddress = value }
        if let value = defaults.string(forKey: DefaultsKey.clientAddress), !value.isEmpty { clientAddress = value }
        if defaults.object(forKey: DefaultsKey.port) != nil { port = defaults.integer(forKey: DefaultsKey.port) }
        if let value = defaults.string(forKey: DefaultsKey.pythonExecutable), !value.isEmpty { pythonExecutable = value }
        if let value = defaults.string(forKey: DefaultsKey.environmentDirectory), !value.isEmpty { environmentDirectory = value }
        if let value = defaults.string(forKey: DefaultsKey.runsDirectory), !value.isEmpty { runsDirectory = value }
        if let value = defaults.string(forKey: DefaultsKey.allowedHosts) { allowedHosts = value }
        bearerAuthenticationEnabled = defaults.bool(forKey: DefaultsKey.bearerAuthentication)
        bearerTokenIsSet = readBearerToken() != nil
    }

    var serverURL: String {
        var components = URLComponents()
        components.scheme = "http"
        components.host = clientAddress.trimmingCharacters(in: .whitespacesAndNewlines)
        components.port = port
        components.path = "/mcp"
        return components.url?.absoluteString ?? "http://127.0.0.1:\(port)/mcp"
    }

    var isLoopbackAddress: Bool {
        Self.isLocalAddress(bindAddress) && Self.isLocalAddress(clientAddress)
    }

    var expandedRunsDirectory: String { Self.expandPath(runsDirectory) }
    var expandedEnvironmentDirectory: String { Self.expandPath(environmentDirectory) }

    func setBearerToken(_ token: String) {
        let value = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { clearBearerToken(); return }
        let data = Data(value.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: KeychainKey.service,
            kSecAttrAccount as String: KeychainKey.account
        ]
        let updates: [String: Any] = [kSecValueData as String: data]
        if SecItemUpdate(query as CFDictionary, updates as CFDictionary) == errSecItemNotFound {
            var item = query
            item[kSecValueData as String] = data
            item[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlocked
            SecItemAdd(item as CFDictionary, nil)
        }
        bearerTokenIsSet = true
    }

    func generateBearerToken() {
        setBearerToken("\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))")
    }

    func clearBearerToken() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: KeychainKey.service,
            kSecAttrAccount as String: KeychainKey.account
        ]
        SecItemDelete(query as CFDictionary)
        bearerTokenIsSet = false
    }

    func authorizationHeaderForCopy() -> String? {
        guard bearerAuthenticationEnabled, let token = readBearerToken() else { return nil }
        return "Authorization: Bearer \(token)"
    }

    func installBackend() {
        guard !isInstalling, !isRunning else { return }
        isInstalling = true
        status = "Preparing the isolated MCP environment…"
        serverOutput = []
        Task {
            defer { isInstalling = false }
            do {
                let python = try Self.resolveExecutable(pythonExecutable)
                let environment = expandedEnvironmentDirectory
                try FileManager.default.createDirectory(
                    at: URL(fileURLWithPath: environment).deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                let create = await Self.runCommand(python, arguments: ["-m", "venv", environment])
                guard create.exitCode == 0 else { throw MCPServerError.commandFailed("Could not create the MCP environment.\n\(create.output)") }
                status = "Installing mlx-lm-lora MCP support…"
                let venvPython = URL(fileURLWithPath: environment).appending(path: "bin/python").path
                let install = await Self.runCommand(venvPython, arguments: ["-m", "pip", "install", "--upgrade", "mlx-lm-lora[mcp]"])
                guard install.exitCode == 0 else { throw MCPServerError.commandFailed("Could not install the MCP backend.\n\(install.output)") }
                status = "MCP backend installed in \(environment)."
                appendOutput(install.output)
            } catch {
                status = error.localizedDescription
            }
        }
    }

    func startServer() {
        guard !isRunning, !isInstalling else { return }
        guard isLoopbackAddress || bearerAuthenticationEnabled else {
            status = "Enable bearer authentication before binding the server beyond this Mac."
            return
        }
        guard !bearerAuthenticationEnabled || bearerTokenIsSet else {
            status = "Generate a bearer token before starting the protected server."
            return
        }

        let environment = expandedEnvironmentDirectory
        let executable = URL(fileURLWithPath: environment).appending(path: "bin/mlx-lm-lora-mcp").path
        guard FileManager.default.isExecutableFile(atPath: executable) else {
            status = "Install the MCP backend first."
            return
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.currentDirectoryURL = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
        var arguments = ["--transport", "http", "--host", bindAddress, "--port", String(port), "--runs-dir", expandedRunsDirectory]
        if !isLoopbackAddress {
            arguments += ["--allowed-host", "\(clientAddress):\(port)"]
        }
        for host in allowedHosts.split(separator: ",").map({ $0.trimmingCharacters(in: .whitespacesAndNewlines) }).filter({ !$0.isEmpty }) {
            arguments += ["--allowed-host", host]
        }
        process.arguments = arguments
        var environmentVariables = ProcessInfo.processInfo.environment
        environmentVariables["MLX_LM_LORA_RUNS_DIR"] = expandedRunsDirectory
        environmentVariables["PYTHONUNBUFFERED"] = "1"
        if bearerAuthenticationEnabled, let token = readBearerToken() {
            environmentVariables["MLX_LM_LORA_MCP_TOKEN"] = token
        }
        process.environment = environmentVariables

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let line = String(decoding: data, as: UTF8.self)
            Task { @MainActor [weak self] in self?.appendOutput(line) }
        }
        process.terminationHandler = { [weak self] process in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.isRunning = false
                self.serverProcess = nil
                self.serverOutputPipe?.fileHandleForReading.readabilityHandler = nil
                self.serverOutputPipe = nil
                self.status = process.terminationStatus == 0
                    ? "MCP server stopped."
                    : "MCP server exited with code \(process.terminationStatus)."
            }
        }

        do {
            try process.run()
            serverProcess = process
            serverOutputPipe = pipe
            isRunning = true
            status = "MCP server running at \(serverURL)"
            serverOutput = []
        } catch {
            status = "Could not start the MCP server: \(error.localizedDescription)"
        }
    }

    func stopServer() {
        guard isRunning else { return }
        status = "Stopping MCP server…"
        serverProcess?.terminate()
    }

    func installClient(_ client: MCPAgentClient) async {
        guard !bearerAuthenticationEnabled else {
            clientStatus[client.rawValue] = "Use the endpoint and bearer token in this client’s MCP settings."
            return
        }
        guard let executable = Self.findClientExecutable(client) else {
            clientStatus[client.rawValue] = "Could not find the \(client.title) CLI. Install it, then try again."
            return
        }

        let url = serverURL
        let addArguments: [String]
        let removeArguments: [String]?
        switch client {
        case .codex:
            addArguments = ["mcp", "add", "mlx-lm-lora", "--url", url]
            removeArguments = ["mcp", "remove", "mlx-lm-lora"]
        case .claudeCode:
            addArguments = ["mcp", "add", "--transport", "http", "--scope", "user", "mlx-lm-lora", url]
            removeArguments = ["mcp", "remove", "mlx-lm-lora", "--scope", "user"]
        case .hermes:
            addArguments = ["mcp", "add", "mlx-lm-lora", "--url", url]
            removeArguments = ["mcp", "remove", "mlx-lm-lora"]
        case .openCode:
            addArguments = ["mcp", "add", "mlx-lm-lora", "--global", "--url", url]
            removeArguments = nil
        }

        clientStatus[client.rawValue] = "Adding MCP connection…"
        if let removeArguments {
            _ = await Self.runCommand(executable, arguments: removeArguments)
        }
        let result = await Self.runCommand(executable, arguments: addArguments)
        if result.exitCode == 0 {
            clientStatus[client.rawValue] = "Connected to \(url). Restart or refresh \(client.title) if it is already open."
        } else {
            clientStatus[client.rawValue] = "Could not configure \(client.title). \(result.output.trimmingCharacters(in: .whitespacesAndNewlines))"
        }
    }

    func setSkillStatus(_ message: String, for key: String) {
        skillStatus[key] = message
    }

    private func appendOutput(_ value: String) {
        let lines = value.split(whereSeparator: \.isNewline).map(String.init)
        serverOutput.append(contentsOf: lines)
        if serverOutput.count > 300 { serverOutput.removeFirst(serverOutput.count - 300) }
    }

    private func readBearerToken() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: KeychainKey.service,
            kSecAttrAccount as String: KeychainKey.account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var item: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private static func expandPath(_ path: String) -> String {
        NSString(string: path).expandingTildeInPath
    }

    private static func isLocalAddress(_ address: String) -> Bool {
        let host = address.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return host == "localhost" || host == "127.0.0.1" || host == "::1"
    }

    private static func findClientExecutable(_ client: MCPAgentClient) -> String? {
        let home = NSHomeDirectory()
        var directories = (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":").map(String.init)
        directories += ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "\(home)/.local/bin", "\(home)/bin", "\(home)/.cargo/bin", "\(home)/.asdf/shims", "\(home)/.local/share/mise/shims", "\(home)/.bun/bin", "\(home)/.npm-global/bin"]
        for base in ["\(home)/.nvm/versions/node", "\(home)/.volta/bin"] {
            if let versions = try? FileManager.default.contentsOfDirectory(atPath: base) {
                directories += versions.map { "\(base)/\($0)/bin" }
            }
            directories.append(base)
        }
        for directory in directories {
            for name in client.executableNames {
                let candidate = URL(fileURLWithPath: directory).appending(path: name).path
                if FileManager.default.isExecutableFile(atPath: candidate) { return candidate }
            }
        }
        return nil
    }

    private static func resolveExecutable(_ path: String) throws -> String {
        let expanded = expandPath(path)
        if expanded.contains("/") {
            guard FileManager.default.isExecutableFile(atPath: expanded) else {
                throw MCPServerError.commandFailed("Python executable not found at \(expanded).")
            }
            return expanded
        }
        let directories = (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":").map(String.init)
        for directory in directories {
            let candidate = URL(fileURLWithPath: directory).appending(path: expanded).path
            if FileManager.default.isExecutableFile(atPath: candidate) { return candidate }
        }
        throw MCPServerError.commandFailed("Could not find \(path) on PATH.")
    }

    private static func runCommand(_ executable: String, arguments: [String]) async -> MCPCommandResult {
        await Task.detached(priority: .utility) {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: executable)
            process.arguments = arguments
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe
            do {
                try process.run()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()
                return MCPCommandResult(exitCode: process.terminationStatus, output: String(decoding: data, as: UTF8.self))
            } catch {
                return MCPCommandResult(exitCode: -1, output: error.localizedDescription)
            }
        }.value
    }
}

private struct MCPCommandResult: Sendable {
    let exitCode: Int32
    let output: String
}

private enum MCPServerError: LocalizedError {
    case commandFailed(String)
    var errorDescription: String? {
        if case let .commandFailed(message) = self { return message }
        return nil
    }
}
