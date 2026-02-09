import Foundation
import EventKit

/// Ingests reminders from EventKit → Memory objects.
/// Working ingestor — EventKit reminders are available on iOS 17+.
/// @unchecked Sendable: EKEventStore is thread-safe for read operations.
final class RemindersIngestor: @unchecked Sendable, IngestorProtocol {
    let source: MemorySource = .reminder

    private let eventStore: EKEventStore

    init(eventStore: EKEventStore = EKEventStore()) {
        self.eventStore = eventStore
    }

    func ingest(since: Date) async throws -> [Memory] {
        let calendars = eventStore.calendars(for: .reminder)
        let predicate = eventStore.predicateForIncompleteReminders(
            withDueDateStarting: since,
            ending: nil,
            calendars: calendars
        )

        let reminders = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[EKReminder], Error>) in
            eventStore.fetchReminders(matching: predicate) { reminders in
                if let reminders {
                    continuation.resume(returning: reminders)
                } else {
                    continuation.resume(returning: [])
                }
            }
        }

        return reminders.compactMap { reminder -> Memory? in
            guard let title = reminder.title, !title.isEmpty else { return nil }

            var content = "Reminder: \(title)"
            if let notes = reminder.notes, !notes.isEmpty {
                content += ". \(notes)"
            }
            if let dueDate = reminder.dueDateComponents?.date {
                content += " — due \(dueDate.formatted(.dateTime.month().day()))"
            }

            let memory = Memory(content: content, source: .reminder)
            memory.sourceID = reminder.calendarItemIdentifier
            return memory
        }
    }

    func requestPermissions() async throws -> Bool {
        return try await eventStore.requestFullAccessToReminders()
    }

    var hasPermissions: Bool {
        EKEventStore.authorizationStatus(for: .reminder) == .fullAccess
    }
}
