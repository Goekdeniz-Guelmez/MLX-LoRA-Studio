import Foundation

/// A Python interpreter discovered on the host system.
/// `id` is the canonicalised absolute path so duplicates collapse
/// even when the same interpreter is reached through different
/// symlinks (e.g. `/usr/bin/python3` vs `/usr/local/bin/python3`).
struct PythonEnvironment: Identifiable, Hashable, Codable {
    enum Kind: String, Codable, Hashable {
        case system
        case homebrew
        case pyenv
        case uv
        case conda
        case venv
        case custom

        var displayName: String {
            switch self {
            case .system: "System"
            case .homebrew: "Homebrew"
            case .pyenv: "pyenv"
            case .uv: "uv"
            case .conda: "Conda"
            case .venv: "Venv"
            case .custom: "Custom"
            }
        }
    }

    /// Canonicalised absolute path (symlinks resolved). Acts as the stable id.
    let id: String
    /// Original path as it was discovered or supplied.
    let path: String
    /// Friendly label such as "Conda · mlx-lm" or "Venv · .venv".
    let label: String
    /// Version string parsed from `platform.python_version()` (e.g. "3.11.5").
    let version: String
    let kind: Kind

    /// Best-effort canonical id. Falls back to the raw path if the URL is malformed.
    static func canonicalID(for rawPath: String) -> String {
        let expanded = NSString(string: rawPath).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded)
        let resolved = url.standardizedFileURL.resolvingSymlinksInPath()
        return resolved.path
    }
}
