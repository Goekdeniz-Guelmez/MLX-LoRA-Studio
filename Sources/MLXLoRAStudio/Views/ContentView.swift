import SwiftUI

struct ContentView: View {
    @Bindable var store: AppStore

    var body: some View {
        ZStack {
            if store.decorativeAnimationsEnabled {
                StudioAnimatedBackground()
                    .ignoresSafeArea()
            }

            NavigationSplitView(columnVisibility: $store.columnVisibility) {
                SidebarView(selection: $store.selection, store: store)
                    .navigationSplitViewColumnWidth(min: 200, ideal: 240, max: 320)
            } detail: {
                detailView
                    .id(store.selection)
                    .transition(.opacity.combined(with: .move(edge: .trailing)))
                    .navigationSplitViewColumnWidth(min: 640, ideal: 940)
            }
            .background(.clear)

            if store.showsOnboarding {
                OnboardingTourView(store: store)
                    .transition(.opacity.combined(with: .scale(scale: 0.98)))
                    .zIndex(10)
            }
        }
        .toolbarBackground(.hidden, for: .windowToolbar)
        .animation(.easeInOut(duration: 0.24), value: store.selection)
        .animation(.spring(response: 0.42, dampingFraction: 0.86), value: store.showsOnboarding)
        .animation(.easeInOut(duration: 0.2), value: store.trainingRunner.isRunning)
        .animation(.easeInOut(duration: 0.2), value: store.trainingRunner.isPaused)
        .toolbar {
            if store.selection == .train {
                ToolbarItemGroup(placement: .primaryAction) {
                    Button {
                        store.trainingRunner.stop()
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                    }
                    .disabled(!store.trainingRunner.isRunning)

                    Button {
                        Task { await store.toggleTrainingPlayback() }
                    } label: {
                        Label(playbackTitle, systemImage: playbackSymbol)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!store.trainingRunner.isRunning && !store.canStartSelectedJob)
                }
            }
        }
    }

    @ViewBuilder
    private var detailView: some View {
        switch store.selection {
        case .train:
            TrainingView(store: store)
        case .metrics:
            LiveMetricsView(runner: store.trainingRunner)
        case .synthetic:
            SyntheticDataView(store: store)
        case .upload:
            HFUploadView(store: store)
        case .guide:
            AlgorithmGuideView(config: $store.training)
        case .runs:
            RunsView(store: store)
        case .about:
            AboutView()
        }
    }

    private var playbackTitle: String {
        if store.trainingRunner.isRunning {
            return store.trainingRunner.isPaused ? "Resume" : "Pause"
        }
        return "Run"
    }

    private var playbackSymbol: String {
        if store.trainingRunner.isRunning {
            return store.trainingRunner.isPaused ? "play.fill" : "pause.fill"
        }
        return "play.fill"
    }
}
