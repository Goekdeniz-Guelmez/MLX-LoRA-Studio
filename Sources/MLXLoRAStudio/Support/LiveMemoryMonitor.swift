import Darwin
import Foundation

@MainActor
final class LiveMemoryMonitor: ObservableObject {
    struct Snapshot: Equatable {
        let chipLabel: String
        let usedMemory: UInt64
        let totalMemory: UInt64
        let sampledAt: Date

        var usedRatio: Double {
            guard totalMemory > 0 else { return 0 }
            return Double(usedMemory) / Double(totalMemory)
        }

        var usedMemoryString: String {
            ByteCountFormatter.string(fromByteCount: Int64(usedMemory), countStyle: .memory)
        }

        var totalMemoryString: String {
            ByteCountFormatter.string(fromByteCount: Int64(totalMemory), countStyle: .memory)
        }
    }

    @Published private(set) var snapshot: Snapshot

    private var timer: Timer?

    init() {
        snapshot = Self.sample()
        start()
    }

    func refresh() {
        snapshot = Self.sample()
    }

    private func start() {
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.refresh()
            }
        }
    }

    private static func sample() -> Snapshot {
        let totalMemory = sysctlU64(name: "hw.memsize") ?? 0
        let chipLabel = humaniseChip(brand: sysctlString(name: "machdep.cpu.brand_string") ?? "Apple silicon")
        return Snapshot(
            chipLabel: chipLabel,
            usedMemory: activeMemoryBytes(),
            totalMemory: totalMemory,
            sampledAt: Date()
        )
    }

    private static func activeMemoryBytes() -> UInt64 {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }

        var hostPageSize: vm_size_t = 0
        guard host_page_size(mach_host_self(), &hostPageSize) == KERN_SUCCESS else { return 0 }
        let pageSize = UInt64(hostPageSize)
        let active = UInt64(stats.active_count)
        let wired = UInt64(stats.wire_count)
        let compressed = UInt64(stats.compressor_page_count)
        return (active + wired + compressed) * pageSize
    }

    private static func humaniseChip(brand: String) -> String {
        let trimmed = brand
            .replacingOccurrences(of: "Apple ", with: "")
            .trimmingCharacters(in: .whitespaces)
        return trimmed.isEmpty ? "Apple silicon" : trimmed
    }

    private static func sysctlString(name: String) -> String? {
        var size: size_t = 0
        sysctlbyname(name, nil, &size, nil, 0)
        guard size > 0 else { return nil }
        var buffer = [UInt8](repeating: 0, count: size)
        let result = sysctlbyname(name, &buffer, &size, nil, 0)
        guard result == 0 else { return nil }
        let bytes = buffer.prefix { $0 != 0 }
        return String(decoding: bytes, as: UTF8.self)
    }

    private static func sysctlU64(name: String) -> UInt64? {
        var value: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        let result = sysctlbyname(name, &value, &size, nil, 0)
        guard result == 0 else { return nil }
        return value
    }
}
