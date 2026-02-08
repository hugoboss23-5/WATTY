import Foundation

/// Semantic search engine — the core of "Hey Siri, what was that idea about..."
/// Queries the embedding index and enriches results with full memory context.
final class RecallEngine {
    private let memoryStore: MemoryStore
    private let embedder: Embedder
    private let embeddingIndex: EmbeddingIndex
    private let similaritySearch: SimilaritySearch

    init(
        memoryStore: MemoryStore = .shared,
        embedder: Embedder = Embedder(),
        embeddingIndex: EmbeddingIndex = EmbeddingIndex()
    ) {
        self.memoryStore = memoryStore
        self.embedder = embedder
        self.embeddingIndex = embeddingIndex
        self.similaritySearch = SimilaritySearch(embedder: embedder, index: embeddingIndex)
    }

    /// Search across all memories by semantic meaning.
    /// Returns full Memory objects ranked by relevance.
    @MainActor
    func search(
        query: String,
        limit: Int = 5,
        sourceFilter: MemorySource? = nil
    ) async throws -> [RecallResult] {
        // Get similarity-ranked memory IDs
        let searchResults = similaritySearch.search(query: query, limit: limit * 2)

        guard !searchResults.isEmpty else { return [] }

        // Fetch full memory objects
        let allMemories = try memoryStore.fetchAllMemories()
        let memoryMap = Dictionary(uniqueKeysWithValues: allMemories.map { ($0.id, $0) })

        // Fetch clusters for context
        let allClusters = try memoryStore.fetchAllClusters()
        let clusterMap = Dictionary(uniqueKeysWithValues: allClusters.map { ($0.id, $0) })

        var results: [RecallResult] = []

        for (memoryID, score) in searchResults {
            guard let memory = memoryMap[memoryID] else { continue }

            // Apply source filter if specified
            if let sourceFilter, memory.source != sourceFilter {
                continue
            }

            let clusterName = memory.clusterID.flatMap { clusterMap[$0]?.name }

            results.append(RecallResult(
                memory: memory,
                relevanceScore: score,
                clusterName: clusterName
            ))

            if results.count >= limit { break }
        }

        // Boost importance of recalled memories (they matter more if you're looking for them)
        for result in results {
            result.memory.importance = min(1.0, result.memory.importance + 0.1)
            result.memory.updatedAt = Date()
        }

        return results
    }

    /// Quick check: how many total memories are indexed?
    @MainActor
    func totalMemoryCount() throws -> Int {
        return try memoryStore.totalMemoryCount()
    }
}

/// A search result with full memory context and relevance score.
struct RecallResult {
    let memory: Memory
    let relevanceScore: Float
    let clusterName: String?
}
