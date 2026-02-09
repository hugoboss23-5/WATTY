import Foundation
import EventKit

/// Ingests calendar events from EventKit → Memory objects.
/// This is a working ingestor — EventKit is available on iOS 17+.
/// @unchecked Sendable: EKEventStore is thread-safe for read operations.
final class CalendarIngestor: @unchecked Sendable, IngestorProtocol {
    let source: MemorySource = .calendar

    private let eventStore: EKEventStore

    init(eventStore: EKEventStore = EKEventStore()) {
        self.eventStore = eventStore
    }

    func ingest(since: Date) async throws -> [Memory] {
        let now = Date()
        // Look ahead 7 days for upcoming events too
        let futureDate = Calendar.current.date(byAdding: .day, value: 7, to: now)!
        let endDate = max(now, futureDate)

        let predicate = eventStore.predicateForEvents(
            withStart: since,
            end: endDate,
            calendars: nil
        )

        let events = eventStore.events(matching: predicate)
        return events.compactMap { event -> Memory? in
            guard let title = event.title, !title.isEmpty else { return nil }

            var content = title
            if let location = event.location, !location.isEmpty {
                content += " at \(location)"
            }
            if let notes = event.notes, !notes.isEmpty {
                content += ". \(notes)"
            }

            let startStr = event.startDate.formatted(
                .dateTime.month().day().hour().minute()
            )
            content += " — \(startStr)"

            let memory = Memory(content: content, source: .calendar)
            memory.sourceID = event.eventIdentifier
            memory.calendarEventID = event.eventIdentifier

            // Extract attendee names
            if let attendees = event.attendees {
                memory.contactNames = attendees.compactMap { $0.name }
            }

            return memory
        }
    }

    func requestPermissions() async throws -> Bool {
        return try await eventStore.requestFullAccessToEvents()
    }

    var hasPermissions: Bool {
        EKEventStore.authorizationStatus(for: .event) == .fullAccess
    }
}
