import Foundation
import EventKit
import WattyCore

/// watty_calendar_prep — Get context for your next upcoming meeting.
/// Cross-references attendees, topic, and past interactions.
struct WattyCalendarPrep: MCPTool {
    private let memoryStore = MemoryStore.shared
    private let recallEngine = RecallEngine()
    private let eventStore = EKEventStore()
    private let formatter = TextFormatter()

    var definition: [String: Any] {
        [
            "name": "watty_calendar_prep",
            "description": "Get preparation context for an upcoming meeting. Watty cross-references the attendees, topic, and your past interactions to surface what you need to know before walking in.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "event_title": [
                        "type": "string",
                        "description": "Title of the event. If omitted, uses the next upcoming event."
                    ] as [String: Any]
                ] as [String: Any]
            ] as [String: Any]
        ]
    }

    @MainActor
    func execute(params: JSONRPCParams) async throws -> Any {
        let eventTitle = params.string(for: "event_title")

        // Find the target event
        let now = Date()
        let endOfDay = Calendar.current.date(byAdding: .day, value: 1, to: now)!
        let predicate = eventStore.predicateForEvents(
            withStart: now, end: endOfDay, calendars: nil
        )
        let events = eventStore.events(matching: predicate)
            .filter { !$0.isAllDay }
            .sorted { $0.startDate < $1.startDate }

        let targetEvent: EKEvent?
        if let eventTitle {
            targetEvent = events.first { $0.title?.lowercased().contains(eventTitle.lowercased()) ?? false }
        } else {
            targetEvent = events.first
        }

        guard let event = targetEvent else {
            return [
                "prep": [
                    "formatted_text": "no upcoming meetings found."
                ] as [String: Any]
            ] as [String: Any]
        }

        let minutesUntil = Int(event.startDate.timeIntervalSince(now) / 60)
        let attendeeNames = event.attendees?.compactMap { $0.name } ?? []

        // Get context for each attendee
        var attendeeContexts: [[String: Any]] = []
        for name in attendeeNames {
            let profile = try memoryStore.fetchContactProfile(name: name)
            attendeeContexts.append([
                "name": name,
                "last_interaction": profile?.lastInteraction.map {
                    ISO8601DateFormatter().string(from: $0)
                } ?? "unknown",
                "last_topics": profile?.recentTopics ?? [],
                "relevant_memories": 0
            ] as [String: Any])
        }

        // Semantic search for related memories
        let query = ([event.title ?? ""] + attendeeNames).joined(separator: " ")
        let relatedMemories = try await recallEngine.search(query: query, limit: 5)
        let relevantNotes = relatedMemories.map { $0.memory.content }

        // Format the prep text
        let formattedText = formatter.formatMeetingPrep(
            eventTitle: event.title ?? "meeting",
            minutesUntil: minutesUntil,
            attendeeContext: attendeeNames.map { name in
                let profile = try? memoryStore.fetchContactProfile(name: name)
                return (name: name, lastTopics: profile?.recentTopics ?? [])
            },
            relevantNotes: relevantNotes,
            talkingPoints: []
        )

        return [
            "prep": [
                "event": event.title ?? "Meeting",
                "starts_in_minutes": minutesUntil,
                "attendee_context": attendeeContexts,
                "relevant_notes": relevantNotes,
                "formatted_text": formattedText
            ] as [String: Any]
        ] as [String: Any]
    }
}
