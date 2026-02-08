import SwiftData
import Foundation

/// Core data model for a single memory — anything Watty has learned about you.
@Model
final class Memory {
    // Identity
    var id: UUID
    var createdAt: Date
    var updatedAt: Date

    // Content
    var content: String
    var source: MemorySource
    var sourceID: String?

    // Embedding — 512-dim NLEmbedding vector
    var embedding: [Float]?

    // Classification
    var clusterID: UUID?
    var importance: Float  // 0.0–1.0, decays over time, boosted by recall

    // Relationships
    var contactNames: [String]
    var calendarEventID: String?

    init(content: String, source: MemorySource) {
        self.id = UUID()
        self.createdAt = Date()
        self.updatedAt = Date()
        self.content = content
        self.source = source
        self.importance = 0.5
        self.contactNames = []
    }
}

enum MemorySource: String, Codable {
    case calendar
    case message
    case mail
    case note
    case reminder
    case contact
    case manual
    case aiConversation
}
