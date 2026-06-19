import Foundation
import AppKit

/// Discovers Hugging Face assets that are already in the local HF cache
/// (`~/.cache/huggingface/hub/`) and offers a small UI for picking
/// between them, adding a new HF repo, or pointing at a local path.
///
/// The HF cache layout is well-defined: each entry is a directory whose
/// name starts with `models--` (a base model) or `datasets--` (a
/// dataset) and follows the convention `type--owner--name`. We use that
/// naming convention directly rather than parsing `refs/main` because
/// the directory name is always the canonical HF id we want to feed
/// back to the trainer.
struct HFCachedAsset: Identifiable, Hashable {
    enum Kind: String { case model, dataset }

    /// Canonical Hugging Face id (`owner/name`). This is the value that
    /// the trainer wants in its `--model` / `--data` spec.
    let hfID: String
    let kind: Kind
    /// Last-modified time of the directory, used to sort "most recently
    /// used" first.
    let modifiedAt: Date

    var id: String { "\(kind.rawValue):\(hfID)" }
}

enum HFCacheScanner {
    /// Returns the user's HF cache root, honouring `HF_HOME` and `HF_HUB_CACHE`
    /// if set, otherwise the standard `~/.cache/huggingface/`.
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

    /// Scan the cache and return all models and datasets. Sorted by
    /// last-modified time, newest first.
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

    /// Convert a cache directory name like `models--mlx-community--Qwen3-0.6B`
    /// into the canonical HF id `mlx-community/Qwen3-0.6B`. Returns nil if
    /// the prefix is missing or the remainder does not have the expected
    /// `owner--name` shape.
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
