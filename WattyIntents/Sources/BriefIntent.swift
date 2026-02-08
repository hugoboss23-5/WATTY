import AppIntents
import EventKit
import WattyCore

/// "Hey Siri, what's my day look like?"
/// Generates and speaks today's daily brief.
struct BriefIntent: AppIntent {
    static let title: LocalizedStringResource = "Watty Brief"
    static let description: IntentDescription = "Get your daily brief"
    static let openAppWhenRun = false

    func perform() async throws -> some IntentResult & ProvidesDialog {
        let generator = BriefGenerator(
            memoryStore: MemoryStore.shared,
            embedder: Embedder(),
            eventStore: EKEventStore()
        )

        let brief = try await generator.generate(for: Date())
        return .result(dialog: "\(brief.formattedText)")
    }
}
