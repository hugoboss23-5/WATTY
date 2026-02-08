import Foundation

/// STUB: Ingests messages via App Intents personal context.
/// Requires iOS 26.4+ with personal context API. Interface is defined
/// so the architecture is ready when Apple ships it.
final class MessageIngestor: IngestorProtocol {
    let source: MemorySource = .message

    func ingest(since: Date) async throws -> [Memory] {
        // STUB — App Intents personal context for Messages
        // not available until iOS 26.4+
        //
        // When available, this will:
        // 1. Use AppIntents personal context to access recent messages
        // 2. Extract text content and sender/recipient
        // 3. Run CommitmentExtractor on message threads
        // 4. Return Memory objects with contact associations
        return []
    }

    func requestPermissions() async throws -> Bool {
        // Personal context permissions are handled by App Intents framework
        return false
    }

    var hasPermissions: Bool {
        // Will return true when iOS 26.4+ personal context is available
        return false
    }
}
