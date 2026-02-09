import Foundation
import NaturalLanguage

/// On-device sentence embedding using Apple's NLEmbedding.
/// 512 dimensions, runs on Neural Engine. No network required.
/// @unchecked Sendable: NLEmbedding is immutable after init and thread-safe for reads.
final class Embedder: @unchecked Sendable {
    private let embedding: NLEmbedding?

    init() {
        self.embedding = NLEmbedding.sentenceEmbedding(for: .english)
    }

    /// Embed a single string → 512-dim vector.
    func embed(_ text: String) -> [Float]? {
        guard let embedding else { return nil }
        guard let vector = embedding.vector(for: text) else { return nil }
        return vector.map { Float($0) }
    }

    /// Embed multiple strings in batch.
    func embedBatch(_ texts: [String]) -> [[Float]?] {
        return texts.map { embed($0) }
    }

    /// Get the embedding dimension (512 for English sentence embedding).
    var dimension: Int {
        return 512
    }

    /// Check if the embedding model is available on this device.
    var isAvailable: Bool {
        return embedding != nil
    }
}
