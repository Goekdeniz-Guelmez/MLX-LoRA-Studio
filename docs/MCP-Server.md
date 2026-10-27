# MCP Server

The MCP Server page manages the `mlx-lm-lora` MCP backend separately from the trainer environment bundled with MLX LoRA Studio. Install or update the package into its own Python environment, configure the HTTP server, and start or stop it from the page.

## Server settings

- **Bind address and port** control where the HTTP server listens. The default is `127.0.0.1:8765`.
- **Client address** is the host agents use to reach the server. It defaults to `127.0.0.1` and can be set to a LAN address or hostname.
- **MCP environment** is an isolated Python virtual environment used for the backend package.
- **Runs folder** selects where the backend writes training run state and logs. The dashboard reads this folder.
- **Allowed hosts** accepts comma-separated host patterns for clients connecting to a non-loopback address.
- **Bearer authentication** stores its token in the macOS Keychain. Authentication is required when either the bind address or client address is not localhost.

The server uses Streamable HTTP at `http://<client-address>:<port>/mcp`. It can also expose stdio for clients that start local MCP processes; the page shows the installed command and run folder environment variable.

## Connect an agent

The page can add the endpoint with the installed CLIs for Codex, Claude Code, Hermes Agent, and OpenCode. For other MCP clients, use the HTTP URL or configure the installed `mlx-lm-lora-mcp` executable with stdio transport.

The skill installer downloads the full `skills/mlx-lm-lora/` folder from the backend repository. It supports the standard directories for Codex, Claude Code, Hermes Agent, OpenCode, and the shared `~/.agents/skills` directory used by community harnesses.

## Live run dashboard

The dashboard refreshes every two seconds and shows saved MCP runs, status, iteration progress, training and validation loss, adapter location, and the recent training log. It reads the backend's persisted run files, so prior runs remain visible after the server or Studio restarts.
