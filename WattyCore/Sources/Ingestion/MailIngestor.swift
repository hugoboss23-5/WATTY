import Foundation

/// STUB: Ingests mail via App Intents personal context.
/// Requires iOS 26.4+ with personal context API.
final class MailIngestor: IngestorProtocol {
    let source: MemorySource = .mail

    func ingest(since: Date) async throws -> [Memory] {
        // STUB — App Intents personal context for Mail
        // not available until iOS 26.4+
        //
        // When available, this will:
        // 1. Use AppIntents personal context to access recent emails
        // 2. Extract subject, sender, key content
        // 3. Identify action items and commitments
        // 4. Return Memory objects with contact associations
        return []
    }

    func requestPermissions() async throws -> Bool {
        return false
    }

    var hasPermissions: Bool {
        return false
    }
}
