import Testing
import Foundation
@testable import WattyCore

@Suite("MemoryStore Tests")
struct MemoryStoreTests {

    @Test("Save and fetch a memory")
    @MainActor
    func testSaveAndFetch() throws {
        let store = MemoryStore(inMemory: true)
        let memory = Memory(content: "Test memory content", source: .manual)

        try store.save(memory)
        let fetched = try store.fetchAllMemories()

        #expect(fetched.count == 1)
        #expect(fetched.first?.content == "Test memory content")
        #expect(fetched.first?.source == .manual)
    }

    @Test("Fetch memories filtered by source")
    @MainActor
    func testFetchBySource() throws {
        let store = MemoryStore(inMemory: true)

        let m1 = Memory(content: "Calendar event", source: .calendar)
        let m2 = Memory(content: "Manual note", source: .manual)
        let m3 = Memory(content: "AI conversation", source: .aiConversation)

        try store.saveAll([m1, m2, m3])

        let calendarMemories = try store.fetchMemories(source: .calendar)
        #expect(calendarMemories.count == 1)
        #expect(calendarMemories.first?.source == .calendar)
    }

    @Test("Save and fetch commitments")
    @MainActor
    func testCommitments() throws {
        let store = MemoryStore(inMemory: true)
        let commitment = Commitment(
            description: "Send the pitch deck",
            sourceMemoryID: UUID(),
            counterparty: "Rim",
            ownerIsUser: true
        )

        try store.saveCommitment(commitment)
        let fetched = try store.fetchCommitments(status: .open)

        #expect(fetched.count == 1)
        #expect(fetched.first?.commitmentDescription == "Send the pitch deck")
        #expect(fetched.first?.counterparty == "Rim")
        #expect(fetched.first?.ownerIsUser == true)
    }

    @Test("Total memory count")
    @MainActor
    func testTotalCount() throws {
        let store = MemoryStore(inMemory: true)

        try store.saveAll([
            Memory(content: "One", source: .manual),
            Memory(content: "Two", source: .manual),
            Memory(content: "Three", source: .calendar)
        ])

        let count = try store.totalMemoryCount()
        #expect(count == 3)
    }

    @Test("Fetch contact profile by name")
    @MainActor
    func testContactProfile() throws {
        let store = MemoryStore(inMemory: true)
        let profile = ContactProfile(contactIdentifier: "abc123", displayName: "Rim")
        profile.recentTopics = ["research", "methodology"]

        try store.saveContactProfile(profile)
        let fetched = try store.fetchContactProfile(name: "Rim")

        #expect(fetched != nil)
        #expect(fetched?.displayName == "Rim")
        #expect(fetched?.recentTopics == ["research", "methodology"])
    }
}
