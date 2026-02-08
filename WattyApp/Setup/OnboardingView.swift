import SwiftUI
import EventKit
import Contacts
import UserNotifications
import WattyCore

/// ONE SCREEN. ONE BUTTON. "Enable Watty."
///
/// Design principles (per Apple Decision Engine):
/// - Simplicity: One action. No configuration. No choices.
/// - User Proxy: User is slightly distracted, has somewhere to be. Make it instant.
/// - Taste: Black screen, white text, one button. Inevitable.
/// - Ship: This screen IS the onboarding. If the user needs more, we've failed.
///
/// After enabling, the user never opens this app again.
/// The product is a text message, not an interface.
struct OnboardingView: View {
    @State private var isEnabled = false
    @State private var isLoading = false

    var body: some View {
        ZStack {
            // Pure black — per Apple-Level UX, surfaces are neutral,
            // content carries the weight
            Color.black.ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer()

                // Watty wordmark — the 60% dominant element (Information Hierarchy)
                Text("watty")
                    .font(.system(size: 48, weight: .bold, design: .default))
                    .foregroundColor(.white)
                    .padding(.bottom, WattyDesignSystem.Spacing.tight)

                // One line — the 30% supporting context
                Text("your phone finally knows you.")
                    .font(.system(size: 17, weight: .regular))
                    .foregroundColor(.white.opacity(0.4))
                    .padding(.bottom, WattyDesignSystem.Spacing.section)

                if !isEnabled {
                    // THE button — minimum touch target 44px (Apple HIG), generous sizing
                    Button(action: enable) {
                        if isLoading {
                            ProgressView()
                                .tint(.black)
                                .frame(width: 280, height: 56)
                        } else {
                            Text("Enable Watty")
                                .font(.system(size: 18, weight: .semibold))
                                .foregroundColor(.black)
                                .frame(width: 280, height: 56)
                        }
                    }
                    .background(Color.white)
                    .cornerRadius(WattyDesignSystem.Layout.cornerRadius)
                    .pressableButton()
                    .padding(.bottom, 20)

                    // The 10% — three lines explaining what happens. That's it.
                    VStack(alignment: .leading, spacing: WattyDesignSystem.Spacing.tight) {
                        permissionLine("Reads your calendar, messages, and notes")
                        permissionLine("Everything stays on your device")
                        permissionLine("Texts you a daily brief every morning")
                    }
                    .padding(.horizontal, 40)
                    .softAppear(isVisible: true, delay: 0.3)

                } else {
                    // Done state — confirmation with Apple-feel spring animation
                    VStack(spacing: WattyDesignSystem.Spacing.tight) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 48))
                            .foregroundColor(WattyDesignSystem.Colors.success)
                            .transition(.scale.combined(with: .opacity))

                        Text("watty is on.")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(.white)

                        Text("your first brief arrives tomorrow at 7am.\nyou can close this app. you won't need it again.")
                            .font(.system(size: 15))
                            .foregroundColor(.white.opacity(0.4))
                            .multilineTextAlignment(.center)
                    }
                    .transition(.opacity.combined(with: .offset(y: 8)))
                }

                Spacer()
                Spacer()
            }
        }
        .animation(WattyDesignSystem.Motion.sheetSpring, value: isEnabled)
    }

    // MARK: - Private

    private func permissionLine(_ text: String) -> some View {
        HStack(spacing: WattyDesignSystem.Spacing.tight + 2) {
            Circle()
                .fill(Color.white.opacity(0.15))
                .frame(width: 6, height: 6)
            Text(text)
                .font(.system(size: 14))
                .foregroundColor(.white.opacity(0.35))
        }
    }

    private func enable() {
        isLoading = true

        Task {
            // Request ALL permissions in sequence
            // Calendar
            let eventStore = EKEventStore()
            try? await eventStore.requestFullAccessToEvents()
            try? await eventStore.requestFullAccessToReminders()

            // Contacts
            let contactStore = CNContactStore()
            try? await contactStore.requestAccess(for: .contacts)

            // Notifications (for daily brief delivery)
            let notifCenter = UNUserNotificationCenter.current()
            try? await notifCenter.requestAuthorization(options: [.alert, .sound])

            // Register background tasks
            BriefScheduler.shared.registerTasks()

            // Start initial ingestion
            try? await IngestionPipeline.shared.runInitialIngestion()

            // Done — transition to enabled state
            await MainActor.run {
                isLoading = false
                isEnabled = true
            }
        }
    }
}
