import Testing
import Foundation
@testable import WattyCore

@Suite("CommitmentExtractor Tests")
struct CommitmentExtractorTests {

    @Test("Extracts 'I'll' promises")
    func testExtractUserPromise() {
        let extractor = CommitmentExtractor(memoryStore: MemoryStore(inMemory: true))
        let memory = Memory(
            content: "I'll send you the updated pitch by tomorrow",
            source: .message
        )
        memory.contactNames = ["Rim"]

        let commitments = extractor.extractCommitments(from: memory)
        #expect(!commitments.isEmpty)
        #expect(commitments.first?.ownerIsUser == true)
        #expect(commitments.first?.counterparty == "Rim")
    }

    @Test("Extracts 'let me' promises")
    func testExtractLetMe() {
        let extractor = CommitmentExtractor(memoryStore: MemoryStore(inMemory: true))
        let memory = Memory(
            content: "Let me check on that and get back to you.",
            source: .message
        )

        let commitments = extractor.extractCommitments(from: memory)
        #expect(!commitments.isEmpty)
        #expect(commitments.first?.ownerIsUser == true)
    }

    @Test("Extracts due dates from text")
    func testDueDateExtraction() {
        let extractor = CommitmentExtractor(memoryStore: MemoryStore(inMemory: true))
        let memory = Memory(
            content: "I'll finish the report tomorrow.",
            source: .message
        )

        let commitments = extractor.extractCommitments(from: memory)
        #expect(!commitments.isEmpty)
        #expect(commitments.first?.dueDate != nil)
    }

    @Test("Does not extract from non-commitment text")
    func testNoFalsePositives() {
        let extractor = CommitmentExtractor(memoryStore: MemoryStore(inMemory: true))
        let memory = Memory(
            content: "The weather is nice today. Great meeting.",
            source: .message
        )

        let commitments = extractor.extractCommitments(from: memory)
        #expect(commitments.isEmpty)
    }

    @Test("Extracts promises made to user")
    func testOtherPersonPromise() {
        let extractor = CommitmentExtractor(memoryStore: MemoryStore(inMemory: true))
        let memory = Memory(
            content: "He said he'd send the proposal by Friday.",
            source: .message
        )

        let commitments = extractor.extractCommitments(from: memory)
        #expect(!commitments.isEmpty)
        #expect(commitments.first?.ownerIsUser == false)
    }
}
