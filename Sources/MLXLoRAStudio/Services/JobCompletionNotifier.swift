import Foundation
import os
import UserNotifications

enum JobCompletionNotifier {
    private static let logger = Logger(subsystem: "com.goekdeniz.mlx-lora-studio", category: "notifications")

    static var isAvailable: Bool {
        Bundle.main.bundleURL.pathExtension == "app"
            && Bundle.main.bundleIdentifier?.isEmpty == false
    }

    static func requestAuthorizationIfNeeded() async -> Bool {
        guard isAvailable else {
            logger.warning("Notification authorization skipped because app is not running from a bundle. bundleURL=\(Bundle.main.bundleURL.path, privacy: .public)")
            return false
        }
        let center = UNUserNotificationCenter.current()
        let settings = await center.notificationSettings()
        logger.info("Notification authorization status before request: \(settings.authorizationStatus.rawValue, privacy: .public), bundleID=\(Bundle.main.bundleIdentifier ?? "nil", privacy: .public)")
        switch settings.authorizationStatus {
        case .authorized, .provisional, .ephemeral:
            return true
        case .denied:
            logger.warning("Notification authorization was previously denied by macOS.")
            return false
        case .notDetermined:
            do {
                let granted = try await center.requestAuthorization(options: [.alert, .sound])
                logger.info("Notification authorization request completed. granted=\(granted, privacy: .public)")
                return granted
            } catch {
                logger.error("Notification authorization request failed: \(error.localizedDescription, privacy: .public)")
                return false
            }
        @unknown default:
            logger.warning("Notification authorization has unknown status: \(settings.authorizationStatus.rawValue, privacy: .public)")
            return false
        }
    }

    static func send(title: String, body: String) async {
        guard await requestAuthorizationIfNeeded() else { return }

        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default

        let request = UNNotificationRequest(
            identifier: "mlx-lora-studio-\(UUID().uuidString)",
            content: content,
            trigger: nil
        )
        try? await UNUserNotificationCenter.current().add(request)
    }
}
