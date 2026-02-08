import SwiftData
import Foundation

/// A promise extracted from conversations — either you promised or someone promised you.
@Model
final class Commitment {
    var id: UUID
    var createdAt: Date

    // What was promised
    var commitmentDescription: String
    var sourceMemoryID: UUID

    // Who's involved
    var ownerIsUser: Bool   // true = user promised, false = someone promised user
    var counterparty: String

    // Status
    var status: CommitmentStatus
    var dueDate: Date?
    var completedAt: Date?
    var lastRemindedAt: Date?

    init(description: String, sourceMemoryID: UUID, counterparty: String, ownerIsUser: Bool) {
        self.id = UUID()
        self.createdAt = Date()
        self.commitmentDescription = description
        self.sourceMemoryID = sourceMemoryID
        self.counterparty = counterparty
        self.ownerIsUser = ownerIsUser
        self.status = .open
    }
}

enum CommitmentStatus: String, Codable {
    case open
    case completed
    case expired
    case dismissed
}
