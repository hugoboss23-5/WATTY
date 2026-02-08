import SwiftUI
import WattyCore

/// @main entry point — SwiftUI App.
/// Watty has no UI beyond the onboarding screen.
/// The product is a text message, not an app.
///
/// Per the Apple Decision Engine:
/// - Ship: This is the minimum that delivers magic.
/// - Simplicity: One screen. One button. Then invisible.
/// - User Proxy: After setup, you never open this app again.
@main
struct WattyApp: App {
    var body: some Scene {
        WindowGroup {
            OnboardingView()
                .preferredColorScheme(.dark)
        }
    }
}
