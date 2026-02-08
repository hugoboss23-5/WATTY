import Foundation

/// STUB: Ingests notes via App Intents personal context.
/// Requires iOS 26.4+ with personal context API.
final class NotesIngestor: IngestorProtocol {
    let source: MemorySource = .note

    func ingest(since: Date) async throws -> [Memory] {
        // STUB — App Intents personal context for Notes
        // not available until iOS 26.4+
        //
        // When available, this will:
        // 1. Use AppIntents personal context to access recent notes
        // 2. Extract title and body content
        // 3. Identify key ideas and insights
        // 4. Return Memory objects
        return []
    }

    func requestPermissions() async throws -> Bool {
        return false
    }

    var hasPermissions: Bool {
        return false
    }
}
