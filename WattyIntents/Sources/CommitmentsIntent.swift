import AppIntents
import WattyCore

/// "Hey Siri, what did I promise?"
/// Lists open commitments via voice.
struct CommitmentsIntent: AppIntent {
    static let title: LocalizedStringResource = "Watty Commitments"
    static let description: IntentDescription = "See your open commitments"
    static let openAppWhenRun = false

    func perform() async throws -> some IntentResult & ProvidesDialog {
        let store = MemoryStore.shared
        let commitments = try await MainActor.run {
            try store.fetchCommitments(status: .open)
        }

        if commitments.isEmpty {
            return .result(dialog: "you're clear. no open commitments.")
        }

        let response = commitments.prefix(5).map { c in
            let daysOld = Calendar.current.dateComponents(
                [.day], from: c.createdAt, to: Date()
            ).day ?? 0
            if daysOld > 3 {
                return "\(c.counterparty.lowercased()) is waiting — \(c.commitmentDescription.lowercased()). \(daysOld) days."
            } else {
                return "\(c.commitmentDescription.lowercased()) for \(c.counterparty.lowercased())."
            }
        }.joined(separator: "\n")

        return .result(dialog: "\(response)")
    }
}
