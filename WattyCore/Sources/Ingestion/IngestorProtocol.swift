import Foundation

/// Protocol that all data source ingestors implement.
/// Each ingestor converts a specific Apple data source into Memory objects.
protocol IngestorProtocol {
    /// The source identifier for memories created by this ingestor.
    var source: MemorySource { get }

    /// Fetch new data since last ingestion and convert to Memory objects.
    func ingest(since: Date) async throws -> [Memory]

    /// Request necessary system permissions for this data source.
    func requestPermissions() async throws -> Bool

    /// Check if permissions are currently granted.
    var hasPermissions: Bool { get }
}

/// Pipeline coordinator that runs all ingestors and indexes results.
final class IngestionPipeline {
    static let shared = IngestionPipeline()

    private let memoryStore: MemoryStore
    private let embedder: Embedder
    private let embeddingIndex: EmbeddingIndex
    private let similaritySearch: SimilaritySearch

    private var ingestors: [any IngestorProtocol] = []

    init(
        memoryStore: MemoryStore = .shared,
        embedder: Embedder = Embedder(),
        embeddingIndex: EmbeddingIndex = EmbeddingIndex()
    ) {
        self.memoryStore = memoryStore
        self.embedder = embedder
        self.embeddingIndex = embeddingIndex
        self.similaritySearch = SimilaritySearch(embedder: embedder, index: embeddingIndex)

        // Register all ingestors
        self.ingestors = [
            CalendarIngestor(),
            RemindersIngestor(),
            ContactsIngestor(),
            MessageIngestor(),
            MailIngestor(),
            NotesIngestor()
        ]
    }

    /// Run initial ingestion for all sources (on first launch).
    @MainActor
    func runInitialIngestion() async throws {
        let thirtyDaysAgo = Calendar.current.date(byAdding: .day, value: -30, to: Date())!
        try await runIngestion(since: thirtyDaysAgo)
    }

    /// Run incremental ingestion for all sources.
    @MainActor
    func runIngestion(since: Date) async throws {
        for ingestor in ingestors where ingestor.hasPermissions {
            do {
                let memories = try await ingestor.ingest(since: since)
                if !memories.isEmpty {
                    try memoryStore.saveAll(memories)
                    similaritySearch.indexMemories(memories)
                }
            } catch {
                // Log but don't fail — other ingestors should still run
                continue
            }
        }
    }

    /// Request permissions for all ingestors.
    func requestAllPermissions() async -> [MemorySource: Bool] {
        var results: [MemorySource: Bool] = [:]
        for ingestor in ingestors {
            let granted = (try? await ingestor.requestPermissions()) ?? false
            results[ingestor.source] = granted
        }
        return results
    }
}
