import Testing
import Foundation
@testable import WattyCore

@Suite("BriefGenerator Tests")
struct BriefGeneratorTests {

    @Test("Brief generates with empty data")
    @MainActor
    func testEmptyBrief() async throws {
        let store = MemoryStore(inMemory: true)
        let generator = BriefGenerator(
            memoryStore: store,
            embedder: Embedder()
        )

        let brief = try await generator.generate(for: Date())

        #expect(!brief.formattedText.isEmpty)
        #expect(brief.formattedText.contains("morning"))
    }

    @Test("Brief includes commitment information")
    @MainActor
    func testBriefWithCommitments() async throws {
        let store = MemoryStore(inMemory: true)

        // Add a commitment
        let commitment = Commitment(
            description: "Send the updated pitch",
            sourceMemoryID: UUID(),
            counterparty: "Rim",
            ownerIsUser: true
        )
        try store.saveCommitment(commitment)

        let generator = BriefGenerator(
            memoryStore: store,
            embedder: Embedder()
        )

        let brief = try await generator.generate(for: Date())

        #expect(!brief.formattedText.isEmpty)
        #expect(!brief.commitments.isEmpty)
        #expect(brief.commitments.first?.counterparty == "Rim")
    }

    @Test("Brief text is lowercase (Watty's voice)")
    @MainActor
    func testBriefVoice() async throws {
        let store = MemoryStore(inMemory: true)
        let generator = BriefGenerator(
            memoryStore: store,
            embedder: Embedder()
        )

        let brief = try await generator.generate(for: Date())

        // Watty speaks in lowercase
        let lines = brief.formattedText.components(separatedBy: "\n")
        for line in lines where !line.isEmpty {
            let firstChar = line.first!
            #expect(firstChar.isLowercase || !firstChar.isLetter,
                   "Brief should be lowercase, but found: \(line)")
        }
    }

    @Test("TextFormatter produces clear day message when empty")
    func testClearDayMessage() async throws {
        let formatter = TextFormatter()
        let text = try await formatter.formatBrief(
            calendar: nil,
            commitments: [],
            nudges: [],
            preps: [],
            date: Date()
        )

        #expect(text.contains("morning"))
        #expect(text.contains("clear day"))
    }
}
