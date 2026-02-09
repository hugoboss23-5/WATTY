import Foundation
import Accelerate

/// High-level semantic search combining Embedder + EmbeddingIndex.
/// Provides the search interface used by RecallEngine and BriefGenerator.
/// Sendable: all stored properties are Sendable (Embedder, EmbeddingIndex).
final class SimilaritySearch: Sendable {
    private let embedder: Embedder
    private let index: EmbeddingIndex

    init(embedder: Embedder, index: EmbeddingIndex) {
        self.embedder = embedder
        self.index = index
    }

    /// Search for memories similar to a natural language query.
    /// Returns (memoryID, score) pairs sorted by relevance.
    func search(query: String, limit: Int = 5) -> [(memoryID: UUID, score: Float)] {
        guard let queryEmbedding = embedder.embed(query) else {
            return []
        }
        return index.search(query: queryEmbedding, limit: limit)
    }

    /// Index a new memory's content.
    func indexMemory(_ memory: Memory) {
        guard let embedding = embedder.embed(memory.content) else { return }
        index.add(memoryID: memory.id, embedding: embedding)
    }

    /// Index multiple memories in batch.
    func indexMemories(_ memories: [Memory]) {
        var batch: [(memoryID: UUID, embedding: [Float])] = []
        for memory in memories {
            if let embedding = embedder.embed(memory.content) {
                batch.append((memoryID: memory.id, embedding: embedding))
            }
        }
        index.addBatch(batch)
    }

    /// Remove a memory from the index.
    func removeMemory(_ memoryID: UUID) {
        index.remove(memoryID: memoryID)
    }

    /// Cosine similarity between two text strings.
    func similarity(between a: String, and b: String) -> Float {
        guard let vecA = embedder.embed(a),
              let vecB = embedder.embed(b) else { return 0 }
        return cosineSimilarity(vecA, vecB)
    }

    // MARK: - Accelerate-backed cosine similarity

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }

        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0

        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
        vDSP_dotpr(a, 1, a, 1, &normA, vDSP_Length(a.count))
        vDSP_dotpr(b, 1, b, 1, &normB, vDSP_Length(b.count))

        guard normA > 0, normB > 0 else { return 0 }
        return dot / (sqrt(normA) * sqrt(normB))
    }
}
