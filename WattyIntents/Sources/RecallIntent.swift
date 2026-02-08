import AppIntents
import WattyCore

/// "Hey Siri, Watty recall [query]"
/// Semantic search across your life via voice.
struct RecallIntent: AppIntent {
    static let title: LocalizedStringResource = "Watty Recall"
    static let description: IntentDescription = "Search your memory by meaning"
    static let openAppWhenRun = false

    @Parameter(title: "What are you looking for?")
    var query: String

    func perform() async throws -> some IntentResult & ProvidesDialog {
        let engine = RecallEngine(
            memoryStore: MemoryStore.shared,
            embedder: Embedder()
        )

        let results = try await engine.search(query: query, limit: 3)

        if results.isEmpty {
            return .result(dialog: "nothing comes to mind for that. try different words?")
        }

        let formatter = TextFormatter()
        let response = results.map { result in
            formatter.formatRecallResult(result)
        }.joined(separator: "\n\n")

        return .result(dialog: "\(response)")
    }
}
