import Testing
import Foundation
@testable import WattyCore

@Suite("RecallEngine Tests")
struct RecallEngineTests {

    @Test("Search returns results ranked by relevance")
    @MainActor
    func testSearchRelevance() async throws {
        let store = MemoryStore(inMemory: true)
        let embedder = Embedder()
        guard embedder.isAvailable else { return }

        let index = EmbeddingIndex(storageDirectory: nil)

        // Create test memories with embeddings
        let m1 = Memory(content: "Arrow's impossibility theorem maps to 7 irreducible policy dimensions", source: .note)
        let m2 = Memory(content: "Meeting with Chen about research methodology next Tuesday", source: .calendar)
        let m3 = Memory(content: "Grocery list: milk, eggs, bread, butter", source: .note)

        if let e1 = embedder.embed(m1.content) { m1.embedding = e1; index.add(memoryID: m1.id, embedding: e1) }
        if let e2 = embedder.embed(m2.content) { m2.embedding = e2; index.add(memoryID: m2.id, embedding: e2) }
        if let e3 = embedder.embed(m3.content) { m3.embedding = e3; index.add(memoryID: m3.id, embedding: e3) }

        try store.saveAll([m1, m2, m3])

        let engine = RecallEngine(
            memoryStore: store,
            embedder: embedder,
            embeddingIndex: index
        )

        let results = try await engine.search(
            query: "economics and geometric shapes",
            limit: 3
        )

        #expect(!results.isEmpty)
        // The Arrow's theorem memory should rank highest for this query
        #expect(results.first?.memory.content.contains("Arrow") == true)
    }

    @Test("Source filter works correctly")
    @MainActor
    func testSourceFilter() async throws {
        let store = MemoryStore(inMemory: true)
        let embedder = Embedder()
        guard embedder.isAvailable else { return }

        let index = EmbeddingIndex(storageDirectory: nil)

        let m1 = Memory(content: "Important note about project", source: .note)
        let m2 = Memory(content: "Important calendar event about project", source: .calendar)

        if let e1 = embedder.embed(m1.content) { m1.embedding = e1; index.add(memoryID: m1.id, embedding: e1) }
        if let e2 = embedder.embed(m2.content) { m2.embedding = e2; index.add(memoryID: m2.id, embedding: e2) }

        try store.saveAll([m1, m2])

        let engine = RecallEngine(
            memoryStore: store,
            embedder: embedder,
            embeddingIndex: index
        )

        let results = try await engine.search(
            query: "project",
            limit: 5,
            sourceFilter: .note
        )

        #expect(results.count == 1)
        #expect(results.first?.memory.source == .note)
    }

    @Test("Empty query returns no results")
    @MainActor
    func testEmptySearch() async throws {
        let store = MemoryStore(inMemory: true)
        let embedder = Embedder()
        let index = EmbeddingIndex(storageDirectory: nil)

        let engine = RecallEngine(
            memoryStore: store,
            embedder: embedder,
            embeddingIndex: index
        )

        let results = try await engine.search(query: "anything", limit: 5)
        #expect(results.isEmpty)
    }
}
