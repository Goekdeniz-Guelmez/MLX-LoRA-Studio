import Foundation
import AppKit

struct HFCachedAsset: Identifiable, Hashable {
    enum Kind: String { case model, dataset }

    let hfID: String
    let kind: Kind

    let modifiedAt: Date

    var id: String { "\(kind.rawValue):\(hfID)" }
}

enum HFCacheScanner {
    static func cacheRoot(fileManager: FileManager = .default) -> URL {
        let env = ProcessInfo.processInfo.environment
        if let hub = env["HF_HUB_CACHE"], !hub.isEmpty {
            return URL(fileURLWithPath: hub, isDirectory: true)
        }
        if let home = env["HF_HOME"], !home.isEmpty {
            return URL(fileURLWithPath: home, isDirectory: true).appending(path: "hub", directoryHint: .isDirectory)
        }
        return URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".cache", directoryHint: .isDirectory)
            .appending(path: "huggingface", directoryHint: .isDirectory)
            .appending(path: "hub", directoryHint: .isDirectory)
    }

    static func scan(fileManager: FileManager = .default) -> (models: [HFCachedAsset], datasets: [HFCachedAsset]) {
        let root = cacheRoot(fileManager: fileManager)
        let contents: [URL]
        do {
            contents = try fileManager.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
            )
        } catch {
            return ([], [])
        }

        var models: [HFCachedAsset] = []
        var datasets: [HFCachedAsset] = []

        for url in contents {
            let name = url.lastPathComponent
            let modified = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? .distantPast

            if name.hasPrefix("models--") {
                if let id = stripPrefix(name, prefix: "models--") {
                    models.append(HFCachedAsset(hfID: id, kind: .model, modifiedAt: modified))
                }
            } else if name.hasPrefix("datasets--") {
                if let id = stripPrefix(name, prefix: "datasets--") {
                    datasets.append(HFCachedAsset(hfID: id, kind: .dataset, modifiedAt: modified))
                }
            }
        }

        let sortByRecent: (HFCachedAsset, HFCachedAsset) -> Bool = { $0.modifiedAt > $1.modifiedAt }
        return (models.sorted(by: sortByRecent), datasets.sorted(by: sortByRecent))
    }
    
    private static func stripPrefix(_ name: String, prefix: String) -> String? {
        guard name.hasPrefix(prefix) else { return nil }
        let rest = String(name.dropFirst(prefix.count))
        let parts = rest.components(separatedBy: "--")
        guard parts.count >= 2 else { return nil }
        let owner = parts[0]
        let remainder = parts.dropFirst().joined(separator: "--")
        guard !owner.isEmpty, !remainder.isEmpty else { return nil }
        return "\(owner)/\(remainder)"
    }
}
