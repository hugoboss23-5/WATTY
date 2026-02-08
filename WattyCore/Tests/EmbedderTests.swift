import Testing
@testable import WattyCore

@Suite("Embedder Tests")
struct EmbedderTests {

    @Test("Embedder initializes and reports availability")
    func testEmbedderInit() {
        let embedder = Embedder()
        // On macOS/iOS with NLEmbedding available, this should be true
        // In CI without the model, it may be false — either is valid
        #expect(embedder.dimension == 512)
    }

    @Test("Embedding produces correct dimensionality")
    func testEmbeddingDimension() {
        let embedder = Embedder()
        guard embedder.isAvailable else { return }

        let vector = embedder.embed("Hello world")
        #expect(vector != nil)
        #expect(vector?.count == 512)
    }

    @Test("Similar texts have higher similarity than dissimilar texts")
    func testSimilarityOrdering() {
        let embedder = Embedder()
        guard embedder.isAvailable else { return }

        let a = embedder.embed("I need to call my mother tomorrow")
        let b = embedder.embed("Remind me to phone mom in the morning")
        let c = embedder.embed("The quick brown fox jumps over the lazy dog")

        guard let a, let b, let c else { return }

        let search = SimilaritySearch(embedder: embedder, index: EmbeddingIndex(storageDirectory: nil))
        // a and b should be more similar to each other than to c
        let simAB = cosineSim(a, b)
        let simAC = cosineSim(a, c)
        #expect(simAB > simAC)
    }

    @Test("Batch embedding works correctly")
    func testBatchEmbedding() {
        let embedder = Embedder()
        guard embedder.isAvailable else { return }

        let texts = ["Hello", "World", "Test"]
        let results = embedder.embedBatch(texts)
        #expect(results.count == 3)
        for result in results {
            #expect(result != nil)
            #expect(result?.count == 512)
        }
    }

    // Helper
    private func cosineSim(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        guard normA > 0, normB > 0 else { return 0 }
        return dot / (sqrt(normA) * sqrt(normB))
    }
}
