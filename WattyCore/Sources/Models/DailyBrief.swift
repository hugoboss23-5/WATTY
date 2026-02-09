import Foundation

/// Structured output model for the daily brief — the core product.
struct DailyBrief: Codable, Sendable {
    let generatedAt: Date
    let date: Date

    // Sections — each is optional, only included if relevant
    let calendarSummary: CalendarSection?
    let commitments: [CommitmentSnapshot]
    let relationshipNudges: [RelationshipNudge]
    let meetingPreps: [MeetingPrep]

    // The formatted text message — this is what the user sees
    let formattedText: String

    struct CalendarSection: Codable {
        let eventCount: Int
        let firstEvent: String
        let busyHours: Int
        let freeBlocks: [String]
    }

    struct CommitmentSnapshot: Codable {
        let id: UUID
        let description: String
        let counterparty: String
        let ownerIsUser: Bool
        let daysOld: Int
        let dueDate: Date?
        let status: String
    }

    struct RelationshipNudge: Codable {
        let contactName: String
        let reason: String
        let urgency: Urgency
    }

    struct MeetingPrep: Codable {
        let eventTitle: String
        let startTime: Date
        let context: String
        let relevantMemories: [UUID]
    }

    enum Urgency: String, Codable {
        case low
        case medium
        case high
    }
}
