import AppIntents

/// Registers all Watty intents with the system for Siri, Shortcuts, and Spotlight.
struct WattyShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: RecallIntent(),
            phrases: [
                "Watty recall \(\.$query)",
                "Watty what do I know about \(\.$query)",
                "Watty find \(\.$query)",
                "Ask Watty about \(\.$query)"
            ],
            shortTitle: "Watty Recall",
            systemImageName: "brain.head.profile"
        )

        AppShortcut(
            intent: BriefIntent(),
            phrases: [
                "Watty brief",
                "What's my day look like",
                "Watty what's today",
                "Morning brief"
            ],
            shortTitle: "Watty Brief",
            systemImageName: "sun.horizon"
        )

        AppShortcut(
            intent: CommitmentsIntent(),
            phrases: [
                "What did I promise",
                "Watty commitments",
                "What do I owe people"
            ],
            shortTitle: "Watty Commitments",
            systemImageName: "checkmark.circle"
        )
    }
}
