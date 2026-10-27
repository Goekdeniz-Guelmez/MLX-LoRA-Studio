import Foundation

enum MCPCommunitySkillInstaller {
    private static let repository = "Goekdeniz-Guelmez/mlx-lm-lora"
    private static let branch = "main"
    private static let skillPrefix = "skills/mlx-lm-lora/"

    static func install(into skillRootPath: String) async throws -> URL {
        let requestURL = URL(string: "https://api.github.com/repos/\(repository)/git/trees/\(branch)?recursive=1")!
        var request = URLRequest(url: requestURL)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("MLX-LoRA-Studio", forHTTPHeaderField: "User-Agent")
        let (treeData, response) = try await URLSession.shared.data(for: request)
        try validate(response)
        let tree = try JSONDecoder().decode(GitTreeResponse.self, from: treeData)
        let files = tree.tree.filter { $0.type == "blob" && $0.path.hasPrefix(skillPrefix) }
        guard files.contains(where: { $0.path == skillPrefix + "SKILL.md" }) else {
            throw MCPCommunitySkillError.missingSkill
        }

        let root = URL(fileURLWithPath: NSString(string: skillRootPath).expandingTildeInPath, isDirectory: true)
        let destination = root.appending(path: "mlx-lm-lora", directoryHint: .isDirectory)
        let staging = root.appending(path: ".mlx-lm-lora-install-\(UUID().uuidString)", directoryHint: .isDirectory)
        let backup = root.appending(path: ".mlx-lm-lora-backup-\(UUID().uuidString)", directoryHint: .isDirectory)
        let manager = FileManager.default
        try manager.createDirectory(at: staging, withIntermediateDirectories: true)

        do {
            for entry in files {
                let relativePath = String(entry.path.dropFirst(skillPrefix.count))
                guard !relativePath.isEmpty,
                      !relativePath.split(separator: "/").contains("..") else { continue }
                let encodedPath = entry.path
                    .split(separator: "/")
                    .map { String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
                    .joined(separator: "/")
                guard let fileURL = URL(string: "https://raw.githubusercontent.com/\(repository)/\(branch)/\(encodedPath)") else {
                    throw MCPCommunitySkillError.invalidFileURL(entry.path)
                }
                let (fileData, fileResponse) = try await URLSession.shared.data(from: fileURL)
                try validate(fileResponse)
                let output = staging.appending(path: relativePath)
                try manager.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
                try fileData.write(to: output, options: .atomic)
            }

            try manager.createDirectory(at: root, withIntermediateDirectories: true)
            if manager.fileExists(atPath: destination.path) {
                try manager.moveItem(at: destination, to: backup)
            }
            do {
                try manager.moveItem(at: staging, to: destination)
                if manager.fileExists(atPath: backup.path) { try manager.removeItem(at: backup) }
            } catch {
                if manager.fileExists(atPath: backup.path), !manager.fileExists(atPath: destination.path) {
                    try? manager.moveItem(at: backup, to: destination)
                }
                throw error
            }
            return destination
        } catch {
            try? manager.removeItem(at: staging)
            throw error
        }
    }

    private static func validate(_ response: URLResponse) throws {
        guard let response = response as? HTTPURLResponse,
              (200..<300).contains(response.statusCode) else {
            let status = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw MCPCommunitySkillError.http(status)
        }
    }
}

private struct GitTreeResponse: Decodable {
    let tree: [GitTreeEntry]
}

private struct GitTreeEntry: Decodable {
    let path: String
    let type: String
}

private enum MCPCommunitySkillError: LocalizedError {
    case missingSkill
    case invalidFileURL(String)
    case http(Int)

    var errorDescription: String? {
        switch self {
        case .missingSkill:
            "The mlx-lm-lora repository did not return its skill folder."
        case let .invalidFileURL(path):
            "Could not download skill file \(path)."
        case let .http(code):
            "The skill download returned HTTP \(code)."
        }
    }
}
