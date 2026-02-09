import Foundation
import Accelerate

/// On-disk vector index for fast semantic search using brute-force cosine similarity.
/// Uses Accelerate framework for SIMD performance on Apple Silicon.
///
/// For v1, this is a flat index (brute-force search). At scale (>10k memories),
/// this should be replaced with HNSW approximate nearest neighbors.
/// @unchecked Sendable: thread safety enforced via concurrent DispatchQueue with barrier writes.
final class EmbeddingIndex: @unchecked Sendable {
    /// An indexed entry: memory ID + its embedding vector.
    struct Entry: Codable {
        let memoryID: UUID
        let embedding: [Float]
    }

    private var entries: [Entry] = []
    private let fileURL: URL
    private let queue = DispatchQueue(label: "com.watty.embedding-index", attributes: .concurrent)

    init(storageDirectory: URL? = nil) {
        let dir = storageDirectory ?? FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first!.appendingPathComponent("Watty", isDirectory: true)

        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        self.fileURL = dir.appendingPathComponent("embedding_index.json")
        loadFromDisk()
    }

    // MARK: - Index Operations

    /// Add a single entry to the index.
    func add(memoryID: UUID, embedding: [Float]) {
        queue.async(flags: .barrier) {
            self.entries.append(Entry(memoryID: memoryID, embedding: embedding))
            self.saveToDisk()
        }
    }

    /// Add multiple entries at once.
    func addBatch(_ newEntries: [(memoryID: UUID, embedding: [Float])]) {
        queue.async(flags: .barrier) {
            for entry in newEntries {
                self.entries.append(Entry(memoryID: entry.memoryID, embedding: entry.embedding))
            }
            self.saveToDisk()
        }
    }

    /// Remove an entry by memory ID.
    func remove(memoryID: UUID) {
        queue.async(flags: .barrier) {
            self.entries.removeAll { $0.memoryID == memoryID }
            self.saveToDisk()
        }
    }

    /// Search for the top-k most similar entries to the query embedding.
    func search(query: [Float], limit: Int = 5) -> [(memoryID: UUID, score: Float)] {
        queue.sync {
            guard !entries.isEmpty else { return [] }

            var results: [(memoryID: UUID, score: Float)] = []

            for entry in entries {
                let score = cosineSimilarity(query, entry.embedding)
                results.append((memoryID: entry.memoryID, score: score))
            }

            // Sort by score descending, return top-k
            results.sort { $0.score > $1.score }
            return Array(results.prefix(limit))
        }
    }

    /// Total number of indexed embeddings.
    var count: Int {
        queue.sync { entries.count }
    }

    // MARK: - Cosine Similarity via Accelerate

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

    // MARK: - Persistence

    private func loadFromDisk() {
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return }
        do {
            let data = try Data(contentsOf: fileURL)
            entries = try JSONDecoder().decode([Entry].self, from: data)
        } catch {
            entries = []
        }
    }

    private func saveToDisk() {
        do {
            let data = try JSONEncoder().encode(entries)
            try data.write(to: fileURL, options: .atomic)
        } catch {
            // Silent failure — index will rebuild from MemoryStore on next launch
        }
    }

    /// Rebuild the entire index from scratch (used after data migration or corruption).
    func rebuild(from memoriesWithEmbeddings: [(memoryID: UUID, embedding: [Float])]) {
        queue.async(flags: .barrier) {
            self.entries = memoriesWithEmbeddings.map {
                Entry(memoryID: $0.memoryID, embedding: $0.embedding)
            }
            self.saveToDisk()
        }
    }
}
