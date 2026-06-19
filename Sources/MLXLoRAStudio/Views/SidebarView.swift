import SwiftUI

/// Loads the app logo from the bundle or, during SwiftPM development,
/// directly from `Sources/Media/logo.png`.
enum AppIcon {
    /// Resolves the bundled icon first, then the source-tree logo used
    /// when running from `swift run`.
    static func bundledImage() -> Image? {
        for url in candidateURLs {
            if let nsImage = NSImage(contentsOf: url) {
                return Image(nsImage: nsImage)
            }
        }
        return nil
    }

    private static var candidateURLs: [URL] {
        let bundleURLs = [
            Bundle.main.url(forResource: "AppIcon", withExtension: "png"),
            Bundle.main.url(forResource: "logo", withExtension: "png")
        ].compactMap { $0 }

        let developmentURLs = developmentRoots.map {
            $0.appendingPathComponent("Sources/Media/logo.png")
        }

        return bundleURLs + developmentURLs
    }

    private static var developmentRoots: [URL] {
        let fileManager = FileManager.default
        let currentDirectory = URL(fileURLWithPath: fileManager.currentDirectoryPath)
        let bundleDirectory = Bundle.main.bundleURL.deletingLastPathComponent()

        return [currentDirectory, bundleDirectory]
    }
}

struct SidebarView: View {
    @Binding var selection: SidebarSection
    let store: AppStore
    @StateObject private var memoryMonitor = LiveMemoryMonitor()

    var body: some View {
        // Four vertical layers: the main section list, a live memory card,
        // an About row, and the author credit pinned to the bottom.
        VStack(spacing: 0) {
            List(selection: $selection) {
                Section {
                    BrandHeader()
                        .listRowSeparator(.hidden)
                        .listRowBackground(Color.clear)
                        .listRowInsets(EdgeInsets(top: 10, leading: 8, bottom: 10, trailing: 8))
                }
                Section {
                    ForEach(SidebarSection.primarySections) { section in
                        Label(section.title, systemImage: section.symbol)
                            .tag(section)
                            .listRowBackground(Color.clear)
                    }
                }
            }
            .listStyle(.sidebar)
            .scrollContentBackground(.hidden)
            .background(.clear)

            VStack(spacing: 10) {
                LiveMemoryCard(
                    snapshot: memoryMonitor.snapshot,
                    estimate: MemoryEstimator.estimate(for: store.training)
                )

                if store.trainingRunner.progressTotal != nil {
                    RunProgressBar(runner: store.trainingRunner)
                }

                Button {
                    selection = .about
                } label: {
                    Label(SidebarSection.about.title, systemImage: SidebarSection.about.symbol)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.plain)
                .padding(.vertical, 8)
                .padding(.horizontal, 12)
                .background {
                    if selection == .about {
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(.selection)
                    }
                }

                // Bottom credit: stays put even if the list scrolls.
                Text("Made with ❤️ by Gökdeniz Gülmez")
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.72)
                    .padding(.vertical, 9)
                    .padding(.horizontal, 12)
                    .frame(maxWidth: .infinity)
                    .liquidGlass(cornerRadius: 999)
            }
            .padding(.horizontal, 10)
            .padding(.top, 8)
            .padding(.bottom, 12)
        }
        .liquidGlass(cornerRadius: 18)
        .padding(10)
        .navigationTitle("Studio")
        .background(.clear)
    }
}

private extension SidebarSection {
    static var primarySections: [SidebarSection] {
        allCases.filter { $0 != .about }
    }
}

/// Sidebar header: small bundled logo (or SF Symbol fallback) next
/// to the app name and version. The version is read from the bundle's
/// `CFBundleShortVersionString` / `CFBundleVersion` when present, so
/// it updates automatically as the plist changes.
private struct BrandHeader: View {
    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            Group {
                if let icon = AppIcon.bundledImage() {
                    icon
                        .resizable()
                        .scaledToFit()
                } else {
                    Image(systemName: "cpu.fill")
                        .resizable()
                        .scaledToFit()
                        .padding(4)
                        .foregroundStyle(.tint)
                }
            }
            // The icon image is clipped with the same rounded shape
            // as the tile, otherwise the bitmap (which has square
            // corners by default) leaks past the material background
            // and the tile looks like a hard square with a tinted
            // margin around it.
            .frame(width: 32, height: 32)
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))

            VStack(alignment: .leading, spacing: 1) {
                Text("MLX LoRA Studio")
                    .font(.headline)
                if let versionLine {
                    Text(versionLine)
                        .font(.system(size: 9, weight: .regular))
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }

    private var versionLine: String? {
        // Use CFBundleShortVersionString only — the build number is
        // noise for a casual UI label. The sidebar should just say
        // "v0.1.0", not "v0.1.0 · build 0.1.0".
        let short = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
        if let short, !short.isEmpty { return "v\(short)" }
        return nil
    }
}
