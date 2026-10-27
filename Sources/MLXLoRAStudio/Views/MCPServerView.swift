import SwiftUI
import AppKit

struct MCPServerView: View {
    @Bindable var store: AppStore
    @Bindable private var controller: MCPServerController
    @State private var dashboardRuns: [MCPTrainingRun] = []
    @State private var selectedRunID: String?
    @State private var isRefreshingDashboard = false

    init(store: AppStore) {
        self.store = store
        self._controller = Bindable(wrappedValue: store.mcpServer)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HeaderView(
                    title: "MCP Server",
                    subtitle: "Manage the mlx-lm-lora agent server and training runs",
                    symbol: "point.3.connected.trianglepath.dotted"
                )

                serverPanel
                settingsPanel
                clientPanel
                skillsPanel
                dashboardPanel
            }
            .padding(24)
            .frame(maxWidth: 1200, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .center)
        }
        .navigationTitle("MCP Server")
        .task {
            await refreshDashboard()
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(2))
                guard !Task.isCancelled else { break }
                await refreshDashboard()
            }
        }
    }

    private var serverPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionTitle("Server")
            HStack(spacing: 8) {
                Circle()
                    .fill(controller.isRunning ? .green : .secondary)
                    .frame(width: 9, height: 9)
                Text(controller.isRunning ? "Running" : (controller.isInstalling ? "Installing" : "Stopped"))
                    .font(.headline)
                Spacer()
                Button {
                    controller.installBackend()
                } label: {
                    Label(controller.isInstalling ? "Installing…" : "Install / Update Backend", systemImage: "arrow.down.circle")
                }
                .disabled(controller.isInstalling || controller.isRunning)

                Button {
                    if controller.isRunning { controller.stopServer() }
                    else { controller.startServer() }
                } label: {
                    Label(controller.isRunning ? "Stop Server" : "Start Server", systemImage: controller.isRunning ? "stop.fill" : "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(controller.isInstalling)
            }

            Text(controller.status)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)

            HStack(spacing: 10) {
                Label(controller.serverURL, systemImage: "link")
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                Spacer()
                Button {
                    copy(controller.serverURL)
                } label: {
                    Label("Copy URL", systemImage: "doc.on.doc")
                }
                .buttonStyle(.bordered)
            }

            if !controller.serverOutput.isEmpty {
                Divider()
                ScrollView {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(controller.serverOutput.suffix(80).enumerated()), id: \.offset) { _, line in
                            Text(line)
                                .font(.system(.caption, design: .monospaced))
                                .textSelection(.enabled)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(height: 120)
            }
        }
        .padding(16)
        .liquidGlass(cornerRadius: 16)
    }

    private var settingsPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionTitle("Server Settings")
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 10) {
                GridRow {
                    Text("Bind address")
                    TextField("127.0.0.1", text: $controller.bindAddress)
                        .textFieldStyle(.roundedBorder)
                    Text("Client address")
                    TextField("127.0.0.1", text: $controller.clientAddress)
                        .textFieldStyle(.roundedBorder)
                }
                GridRow {
                    Text("Port")
                    TextField("8765", value: $controller.port, format: .number)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 120)
                    Text("Python")
                    TextField("/path/to/python3", text: $controller.pythonExecutable)
                        .textFieldStyle(.roundedBorder)
                }
                GridRow {
                    Text("MCP environment")
                    TextField("~/.mlx-lm-lora/studio-mcp-env", text: $controller.environmentDirectory)
                        .textFieldStyle(.roundedBorder)
                        .gridCellColumns(3)
                }
                GridRow {
                    Text("Runs folder")
                    TextField("~/.mlx-lm-lora/runs", text: $controller.runsDirectory)
                        .textFieldStyle(.roundedBorder)
                        .gridCellColumns(3)
                }
                GridRow {
                    Text("Allowed hosts")
                    TextField("One host pattern per comma, e.g. 192.168.1.20:*", text: $controller.allowedHosts)
                        .textFieldStyle(.roundedBorder)
                        .gridCellColumns(3)
                }
            }
            .disabled(controller.isRunning || controller.isInstalling)

            Text("The MCP package is installed into its own Python environment, separate from the trainer bundled with Studio. The dashboard reads the server’s saved run files from the runs folder.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Divider()

            Toggle("Require bearer authentication", isOn: $controller.bearerAuthenticationEnabled)
                .disabled(controller.isRunning)
            HStack {
                Label(controller.bearerTokenIsSet ? "Bearer token stored in Keychain" : "No bearer token stored", systemImage: controller.bearerTokenIsSet ? "key.fill" : "key")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Generate Token") { controller.generateBearerToken() }
                    .disabled(controller.isRunning)
                if controller.bearerTokenIsSet {
                    Button("Copy Auth Header") {
                        if let header = controller.authorizationHeaderForCopy() { copy(header) }
                    }
                    .disabled(!controller.bearerAuthenticationEnabled)
                    Button("Clear") { controller.clearBearerToken() }
                        .disabled(controller.isRunning)
                }
            }
            Text("Authentication is required when either the bind address or client address is not localhost. Use the copied Authorization header in an agent’s MCP settings when bearer authentication is enabled.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(16)
        .liquidGlass(cornerRadius: 16)
    }

    private var clientPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionTitle("Connect an Agent")
            Text("Add the running HTTP server to a supported agent. The CLI must already be installed.")
                .font(.callout)
                .foregroundStyle(.secondary)

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 250), alignment: .top)], alignment: .leading, spacing: 10) {
                ForEach(MCPAgentClient.allCases) { client in
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text(client.title).font(.headline)
                            Spacer()
                            Image(systemName: client.symbol)
                                .foregroundStyle(.secondary)
                        }
                        Button {
                            Task { await controller.installClient(client) }
                        } label: {
                            Label("Install MCP Connection", systemImage: "plus.circle")
                        }
                        .disabled(!controller.isRunning)

                        let connectionStatus = controller.clientStatus[client.rawValue] ?? ""
                        if !connectionStatus.isEmpty {
                            Text(connectionStatus)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                }
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Other MCP clients")
                    .font(.headline)
                Text("Streamable HTTP endpoint: \(controller.serverURL)")
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                Text("For clients that launch local servers, use \(controller.expandedEnvironmentDirectory)/bin/mlx-lm-lora-mcp --transport stdio and set MLX_LM_LORA_RUNS_DIR to \(controller.expandedRunsDirectory).")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
        .padding(16)
        .liquidGlass(cornerRadius: 16)
    }

    private var skillsPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionTitle("Install the Training Skill")
            Text("Download the complete mlx-lm-lora skill folder, including its reference files, into an agent’s skills directory.")
                .font(.callout)
                .foregroundStyle(.secondary)

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 250), alignment: .top)], alignment: .leading, spacing: 10) {
                ForEach(MCPAgentClient.allCases) { client in
                    skillButton(title: client.title, path: client.skillRootPath, key: client.rawValue)
                }
                skillButton(title: "Community agents", path: "~/.agents/skills", key: "community")
            }

            Text("You can also copy the skill folder into another tool’s supported skill directory after installation.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .liquidGlass(cornerRadius: 16)
    }

    private var dashboardPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                SectionTitle("Live Run Dashboard")
                Spacer()
                Button {
                    Task { await refreshDashboard() }
                } label: {
                    Label(isRefreshingDashboard ? "Refreshing…" : "Refresh", systemImage: "arrow.clockwise")
                }
                .disabled(isRefreshingDashboard)
            }

            if dashboardRuns.isEmpty {
                ContentUnavailableView(
                    "No MCP Runs Yet",
                    systemImage: "chart.xyaxis.line",
                    description: Text("Runs started by an agent will appear here with live status, loss, progress, and logs.")
                )
                .frame(maxWidth: .infinity, minHeight: 150)
            } else {
                VStack(spacing: 8) {
                    ForEach(dashboardRuns) { run in
                        Button {
                            selectedRunID = run.id
                        } label: {
                            runRow(run, selected: selectedRunID == run.id)
                        }
                        .buttonStyle(.plain)
                    }
                }

                if let selectedRun {
                    Divider()
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            VStack(alignment: .leading, spacing: 3) {
                                Text(selectedRun.title).font(.headline)
                                Text("\(selectedRun.status.capitalized) · \(selectedRun.id)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                            }
                            Spacer()
                            if !selectedRun.adapterPath.isEmpty {
                                InfoPill(text: "Adapters", symbol: "folder", openPath: selectedRun.adapterPath)
                            }
                        }
                        if !selectedRun.logTail.isEmpty {
                            ScrollView {
                                VStack(alignment: .leading, spacing: 2) {
                                    ForEach(Array(selectedRun.logTail.split(whereSeparator: \.isNewline).suffix(100).enumerated()), id: \.offset) { _, line in
                                        Text(String(line))
                                            .font(.system(.caption, design: .monospaced))
                                            .textSelection(.enabled)
                                    }
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .frame(height: 220)
                            .padding(8)
                            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                        }
                    }
                }
            }
        }
        .padding(16)
        .liquidGlass(cornerRadius: 16)
    }

    private func runRow(_ run: MCPTrainingRun, selected: Bool) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(run.title)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                Spacer()
                Text(run.status.capitalized)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(run.isActive ? .green : .secondary)
            }
            if let progress = run.progress {
                ProgressView(value: progress)
            }
            HStack {
                Text(run.metricsSummary)
                Spacer(minLength: 10)
                Text(run.id)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(.secondary)
        }
        .padding(10)
        .background(selected ? Color.accentColor.opacity(0.13) : Color.clear, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(selected ? Color.accentColor.opacity(0.55) : Color.secondary.opacity(0.14), lineWidth: 1)
        }
    }

    private var selectedRun: MCPTrainingRun? {
        if let selectedRunID, let run = dashboardRuns.first(where: { $0.id == selectedRunID }) { return run }
        return dashboardRuns.first
    }

    private func skillButton(title: String, path: String, key: String) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title).font(.headline)
            Button {
                Task { await installSkill(path: path, key: key) }
            } label: {
                Label("Install Skill", systemImage: "arrow.down.doc")
            }
            let currentStatus = controller.skillStatus[key] ?? ""
            if !currentStatus.isEmpty {
                Text(currentStatus)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    @MainActor
    private func refreshDashboard() async {
        isRefreshingDashboard = true
        defer { isRefreshingDashboard = false }
        let path = controller.expandedRunsDirectory
        dashboardRuns = await Task.detached(priority: .utility) {
            MCPRunDashboardReader.read(from: path)
        }.value
        if let selectedRunID, !dashboardRuns.contains(where: { $0.id == selectedRunID }) {
            self.selectedRunID = nil
        }
    }

    @MainActor
    private func installSkill(path: String, key: String) async {
        controller.setSkillStatus("Downloading skill…", for: key)
        do {
            let installedPath = try await MCPCommunitySkillInstaller.install(into: path)
            controller.setSkillStatus("Installed in \(installedPath.path)", for: key)
        } catch {
            controller.setSkillStatus("Could not install skill: \(error.localizedDescription)", for: key)
        }
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}
