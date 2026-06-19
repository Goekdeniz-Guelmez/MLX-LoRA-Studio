import SwiftUI

struct OnboardingTourView: View {
    @Bindable var store: AppStore
    @State private var stepIndex = 0
    @State private var pulse = false

    private let steps = OnboardingStep.all

    var body: some View {
        ZStack {
            Color.black.opacity(0.24)
                .ignoresSafeArea()

            GeometryReader { proxy in
                spotlightLayer(in: proxy.size)

                VStack {
                    Spacer()
                    tourCard
                        .frame(width: min(560, proxy.size.width - 40))
                        .padding(.bottom, 34)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .onAppear {
            pulse = true
            store.columnVisibility = .all
            store.selection = currentStep.section
        }
        .onChange(of: stepIndex) { _, _ in
            withAnimation(.easeInOut(duration: 0.28)) {
                store.selection = currentStep.section
            }
        }
    }

    private var currentStep: OnboardingStep {
        steps[stepIndex]
    }

    private var isLastStep: Bool {
        stepIndex == steps.count - 1
    }

    private var tourCard: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top, spacing: 14) {
                Image(systemName: currentStep.symbol)
                    .font(.title2.weight(.semibold))
                    .frame(width: 46, height: 46)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))

                VStack(alignment: .leading, spacing: 5) {
                    Text(currentStep.title)
                        .font(.title2.bold())
                    Text(currentStep.message)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Spacer(minLength: 10)

                Button {
                    store.completeOnboarding()
                } label: {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.borderless)
                .help("Skip tour")
            }

            HStack(spacing: 8) {
                ForEach(steps.indices, id: \.self) { index in
                    Capsule()
                        .fill(index == stepIndex ? Color.accentColor : Color.secondary.opacity(0.28))
                        .frame(width: index == stepIndex ? 28 : 8, height: 8)
                        .animation(.spring(response: 0.32, dampingFraction: 0.78), value: stepIndex)
                }
            }
            .accessibilityHidden(true)

            HStack {
                Button("Skip tour") {
                    store.completeOnboarding()
                }

                Spacer()

                Button("Back") {
                    withAnimation(.easeInOut(duration: 0.25)) {
                        stepIndex = max(stepIndex - 1, 0)
                    }
                }
                .disabled(stepIndex == 0)

                Button(isLastStep ? "Finish" : "Next") {
                    advance()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(22)
        .liquidGlass(cornerRadius: 18)
        .shadow(color: .black.opacity(0.18), radius: 22, y: 12)
    }

    @ViewBuilder
    private func spotlightLayer(in size: CGSize) -> some View {
        let rect = currentStep.rect(size)

        RoundedRectangle(cornerRadius: currentStep.cornerRadius, style: .continuous)
            .strokeBorder(Color.accentColor.opacity(0.82), lineWidth: 2)
            .background(
                RoundedRectangle(cornerRadius: currentStep.cornerRadius, style: .continuous)
                    .fill(Color.accentColor.opacity(0.10))
            )
            .shadow(color: Color.accentColor.opacity(pulse ? 0.34 : 0.12), radius: pulse ? 28 : 10)
            .scaleEffect(pulse ? 1.012 : 0.992)
            .frame(width: rect.width, height: rect.height)
            .position(x: rect.midX, y: rect.midY)
            .animation(.easeInOut(duration: 1.15).repeatForever(autoreverses: true), value: pulse)
            .animation(.spring(response: 0.42, dampingFraction: 0.84), value: stepIndex)
            .allowsHitTesting(false)
    }

    private func advance() {
        if isLastStep {
            store.completeOnboarding()
        } else {
            withAnimation(.easeInOut(duration: 0.25)) {
                stepIndex += 1
            }
        }
    }
}

private struct OnboardingStep {
    let title: String
    let message: String
    let symbol: String
    let section: SidebarSection
    let cornerRadius: CGFloat
    let rect: (CGSize) -> CGRect

    @MainActor static let all: [OnboardingStep] = [
        OnboardingStep(
            title: "Choose a workspace",
            message: "The sidebar is your map: train adapters, watch metrics, generate datasets, upload to Hugging Face, read the guide, and revisit saved runs.",
            symbol: "sidebar.left",
            section: .train,
            cornerRadius: 18,
            rect: { size in
                CGRect(x: 10, y: 10, width: min(270, size.width * 0.34), height: max(size.height - 20, 0))
            }
        ),
        OnboardingStep(
            title: "Configure training",
            message: "Start on Train to pick a model, dataset, LoRA settings, and algorithm. The run button in the toolbar launches the selected job.",
            symbol: "cpu",
            section: .train,
            cornerRadius: 18,
            rect: { size in
                let sidebarWidth = min(280, size.width * 0.34)
                return CGRect(x: sidebarWidth + 28, y: 72, width: max(size.width - sidebarWidth - 56, 240), height: max(size.height - 170, 220))
            }
        ),
        OnboardingStep(
            title: "Watch runs live",
            message: "Live Metrics turns training logs into loss, reward, KL, and learning-rate charts while a job is running.",
            symbol: "chart.line.uptrend.xyaxis",
            section: .metrics,
            cornerRadius: 18,
            rect: { size in
                let sidebarWidth = min(280, size.width * 0.34)
                return CGRect(x: sidebarWidth + 34, y: 82, width: max(size.width - sidebarWidth - 68, 240), height: max(size.height - 190, 220))
            }
        ),
        OnboardingStep(
            title: "Create data when you need it",
            message: "Synthetic Data helps generate SFT or preference examples, then Runs keeps local outputs organized so you can inspect or resume later.",
            symbol: "sparkles",
            section: .synthetic,
            cornerRadius: 18,
            rect: { size in
                let sidebarWidth = min(280, size.width * 0.34)
                return CGRect(x: sidebarWidth + 34, y: 82, width: max(size.width - sidebarWidth - 68, 240), height: max(size.height - 190, 220))
            }
        ),
        OnboardingStep(
            title: "Tune the studio",
            message: "Open Settings for Python environments, output folders, Hugging Face tokens, backend package updates, notifications, and resource limits.",
            symbol: "gearshape",
            section: .about,
            cornerRadius: 14,
            rect: { size in
                CGRect(x: max(size.width - 260, 20), y: 18, width: min(220, size.width - 40), height: 58)
            }
        )
    ]
}
