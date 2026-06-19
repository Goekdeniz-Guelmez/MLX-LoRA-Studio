import AppKit
import SwiftUI
import UserNotifications

final class AppDelegate: NSObject, NSApplicationDelegate, UNUserNotificationCenterDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        if JobCompletionNotifier.isAvailable {
            UNUserNotificationCenter.current().delegate = self
        }
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        requestNotificationAuthorizationOnLaunch()
    }

    private func requestNotificationAuthorizationOnLaunch() {
        guard JobCompletionNotifier.isAvailable else { return }
        Task {
            try? await Task.sleep(for: .milliseconds(600))
            _ = await JobCompletionNotifier.requestAuthorizationIfNeeded()
        }
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification
    ) async -> UNNotificationPresentationOptions {
        [.banner, .sound]
    }
}