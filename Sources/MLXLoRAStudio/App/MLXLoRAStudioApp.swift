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

@main
struct MLXLoRAStudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var store = AppStore()

    var body: some Scene {
        WindowGroup("MLX LoRA Studio", id: "main") {
            ContentView(store: store)
                .frame(minWidth: 880, minHeight: 600)
        }
        .commands {
            CommandGroup(after: .newItem) {
                Button(commandTitle) {
                    Task { await store.toggleTrainingPlayback() }
                }
                .keyboardShortcut(.return, modifiers: [.command])
                .disabled(!store.trainingRunner.isRunning && !store.canStartSelectedJob)

                Button("Stop Job") {
                    store.trainingRunner.stop()
                }
                .keyboardShortcut(".", modifiers: [.command])
                .disabled(!store.trainingRunner.isRunning)
            }
        }

        Settings {
            SettingsView(store: store)
                .frame(width: 560)
        }
    }

    private var commandTitle: String {
        guard store.trainingRunner.isRunning else {
            return "Start Job"
        }
        return store.trainingRunner.isPaused ? "Resume Job" : "Pause Job"
    }
}
