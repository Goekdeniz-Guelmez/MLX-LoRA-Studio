import Foundation

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

    let id: String
    let path: String
    let label: String
    let version: String
    let kind: Kind

    static func canonicalID(for rawPath: String) -> String {
        let expanded = NSString(string: rawPath).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded)
        let resolved = url.standardizedFileURL.resolvingSymlinksInPath()
        return resolved.path
    }
}
