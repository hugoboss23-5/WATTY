import SwiftData
import Foundation

/// Relationship context for a contact — how you interact, what you discuss, what's pending.
@Model
final class ContactProfile {
    var id: UUID
    var contactIdentifier: String  // CNContact.identifier
    var displayName: String

    // Relationship intelligence
    var lastInteraction: Date?
    var interactionFrequency: Float  // Messages per week average
    var averageResponseTime: TimeInterval?

    // Context
    var recentTopics: [String]
    var sharedCommitments: [UUID]

    // Notes
    var userNotes: String?

    init(contactIdentifier: String, displayName: String) {
        self.id = UUID()
        self.contactIdentifier = contactIdentifier
        self.displayName = displayName
        self.interactionFrequency = 0
        self.recentTopics = []
        self.sharedCommitments = []
    }
}
