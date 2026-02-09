import Foundation
import NaturalLanguage

/// Extracts commitments (promises) from text content using NLP.
/// Identifies phrases like "I'll send you...", "Let me know...", "I promised..."
/// @unchecked Sendable: commitmentPatterns is a let tuple array (tuples aren't
/// automatically Sendable in Swift 6 even when contents are).
final class CommitmentExtractor: @unchecked Sendable {
    private let memoryStore: MemoryStore

    /// Patterns that indicate a commitment was made.
    private let commitmentPatterns: [(pattern: String, ownerIsUser: Bool)] = [
        // User promising something
        ("i('ll| will) .+", true),
        ("i('m going to| am going to) .+", true),
        ("let me .+", true),
        ("i('ll| will) send .+", true),
        ("i('ll| will) call .+", true),
        ("i('ll| will) get back .+", true),
        ("i promised .+", true),
        ("i need to .+", true),
        ("remind me to .+", true),
        ("i should .+", true),
        // Someone promising user something
        ("(he|she|they)('ll| will) .+", false),
        ("(he|she|they) promised .+", false),
        ("(he|she|they) said (he|she|they)('d|'ll| would| will) .+", false),
    ]

    init(memoryStore: MemoryStore = .shared) {
        self.memoryStore = memoryStore
    }

    /// Extract commitments from a memory's content.
    func extractCommitments(from memory: Memory) -> [Commitment] {
        let text = memory.content.lowercased()
        let sentences = splitIntoSentences(text)

        var commitments: [Commitment] = []

        for sentence in sentences {
            for (pattern, ownerIsUser) in commitmentPatterns {
                if matchesPattern(sentence, pattern: pattern) {
                    let counterparty = extractCounterparty(
                        from: sentence,
                        contactNames: memory.contactNames
                    )

                    let commitment = Commitment(
                        description: cleanCommitmentDescription(sentence),
                        sourceMemoryID: memory.id,
                        counterparty: counterparty,
                        ownerIsUser: ownerIsUser
                    )

                    // Try to extract a due date
                    commitment.dueDate = extractDueDate(from: sentence, relativeTo: memory.createdAt)

                    commitments.append(commitment)
                    break // One commitment per sentence
                }
            }
        }

        return commitments
    }

    /// Process all recent memories and extract new commitments.
    @MainActor
    func processRecentMemories(since: Date) async throws -> [Commitment] {
        let memories = try memoryStore.fetchMemories(since: since)
        var allCommitments: [Commitment] = []

        for memory in memories {
            let extracted = extractCommitments(from: memory)
            for commitment in extracted {
                try memoryStore.saveCommitment(commitment)
                allCommitments.append(commitment)
            }
        }

        return allCommitments
    }

    // MARK: - Private Helpers

    private func splitIntoSentences(_ text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        var sentences: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            sentences.append(String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines))
            return true
        }
        return sentences
    }

    private func matchesPattern(_ text: String, pattern: String) -> Bool {
        let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive)
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex?.firstMatch(in: text, range: range) != nil
    }

    private func extractCounterparty(from sentence: String, contactNames: [String]) -> String {
        // Try to match a known contact name
        for name in contactNames {
            if sentence.lowercased().contains(name.lowercased()) {
                return name
            }
        }
        // Default to "someone" if no contact found
        return "someone"
    }

    private func cleanCommitmentDescription(_ sentence: String) -> String {
        var cleaned = sentence
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "^(i('ll| will| need to| should| am going to|'m going to)|let me|remind me to) ",
                                  with: "",
                                  options: .regularExpression)
        // Capitalize first letter
        if let first = cleaned.first {
            cleaned = first.uppercased() + cleaned.dropFirst()
        }
        return cleaned
    }

    private func extractDueDate(from sentence: String, relativeTo baseDate: Date) -> Date? {
        let calendar = Calendar.current
        let lowered = sentence.lowercased()

        if lowered.contains("today") {
            return calendar.startOfDay(for: baseDate)
        } else if lowered.contains("tomorrow") {
            return calendar.date(byAdding: .day, value: 1, to: calendar.startOfDay(for: baseDate))
        } else if lowered.contains("this week") {
            // End of current week
            let weekday = calendar.component(.weekday, from: baseDate)
            let daysUntilSunday = 8 - weekday
            return calendar.date(byAdding: .day, value: daysUntilSunday, to: calendar.startOfDay(for: baseDate))
        } else if lowered.contains("next week") {
            return calendar.date(byAdding: .weekOfYear, value: 1, to: baseDate)
        }

        return nil
    }
}
