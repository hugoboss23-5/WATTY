import Foundation
import EventKit

/// The most important file in the project. This generates the Daily Brief — the product.
/// Cross-references calendar, commitments, contacts, and memories to produce
/// a single text message that tells you everything you need to know about your day.
/// @unchecked Sendable: EKEventStore is thread-safe for reads; all other deps are Sendable.
final class BriefGenerator: @unchecked Sendable {
    private let memoryStore: MemoryStore
    private let embedder: Embedder
    private let embeddingIndex: EmbeddingIndex
    private let eventStore: EKEventStore
    private let formatter: TextFormatter

    init(
        memoryStore: MemoryStore = .shared,
        embedder: Embedder = Embedder(),
        embeddingIndex: EmbeddingIndex = EmbeddingIndex(),
        eventStore: EKEventStore = EKEventStore()
    ) {
        self.memoryStore = memoryStore
        self.embedder = embedder
        self.embeddingIndex = embeddingIndex
        self.eventStore = eventStore
        self.formatter = TextFormatter()
    }

    /// Generate the daily brief for a given date.
    @MainActor
    func generate(for date: Date) async throws -> DailyBrief {
        // 1. CALENDAR — What's on your schedule today?
        let calendar = getCalendarSection(for: date)

        // 2. COMMITMENTS — What did you promise? What's overdue?
        let commitments = try getOpenCommitments(for: date)

        // 3. RELATIONSHIP NUDGES — Who should you reach out to?
        let nudges = try getRelationshipNudges(for: date)

        // 4. MEETING PREP — Any meetings coming up?
        let preps = try await getMeetingPreps(for: date)

        // 5. FORMAT — Generate the text message
        let formattedText = try await formatter.formatBrief(
            calendar: calendar,
            commitments: commitments,
            nudges: nudges,
            preps: preps,
            date: date
        )

        return DailyBrief(
            generatedAt: Date(),
            date: date,
            calendarSummary: calendar,
            commitments: commitments,
            relationshipNudges: nudges,
            meetingPreps: preps,
            formattedText: formattedText
        )
    }

    // MARK: - Sub-generators

    private func getCalendarSection(for date: Date) -> DailyBrief.CalendarSection? {
        let startOfDay = Calendar.current.startOfDay(for: date)
        guard let endOfDay = Calendar.current.date(byAdding: .day, value: 1, to: startOfDay) else {
            return nil
        }

        let predicate = eventStore.predicateForEvents(
            withStart: startOfDay, end: endOfDay, calendars: nil
        )
        let events = eventStore.events(matching: predicate)
            .filter { !$0.isAllDay }
            .sorted { $0.startDate < $1.startDate }

        guard !events.isEmpty else { return nil }

        let freeBlocks = calculateFreeBlocks(events: events, dayStart: startOfDay, dayEnd: endOfDay)

        let first = events[0]
        let firstStr = "\(first.title ?? "Event") at \(formatTime(first.startDate))"

        let busyMinutes = events.reduce(0) { total, event in
            total + Int(event.endDate.timeIntervalSince(event.startDate) / 60)
        }

        return DailyBrief.CalendarSection(
            eventCount: events.count,
            firstEvent: firstStr,
            busyHours: busyMinutes / 60,
            freeBlocks: freeBlocks
        )
    }

    @MainActor
    private func getOpenCommitments(for date: Date) throws -> [DailyBrief.CommitmentSnapshot] {
        let commitments = try memoryStore.fetchCommitments(status: .open)
            .sorted { ($0.dueDate ?? .distantFuture) < ($1.dueDate ?? .distantFuture) }

        return commitments.prefix(5).map { commitment in
            let daysOld = Calendar.current.dateComponents(
                [.day], from: commitment.createdAt, to: date
            ).day ?? 0

            return DailyBrief.CommitmentSnapshot(
                id: commitment.id,
                description: commitment.commitmentDescription,
                counterparty: commitment.counterparty,
                ownerIsUser: commitment.ownerIsUser,
                daysOld: daysOld,
                dueDate: commitment.dueDate,
                status: commitment.status.rawValue
            )
        }
    }

    @MainActor
    private func getRelationshipNudges(for date: Date) throws -> [DailyBrief.RelationshipNudge] {
        let contactCommitments = try memoryStore.fetchCommitments(status: .open)
            .filter {
                let desc = $0.commitmentDescription.lowercased()
                return desc.contains("call") ||
                       desc.contains("text") ||
                       desc.contains("reach out") ||
                       desc.contains("check in") ||
                       desc.contains("follow up")
            }

        return contactCommitments.map { commitment in
            let daysOld = Calendar.current.dateComponents(
                [.day], from: commitment.createdAt, to: date
            ).day ?? 0
            let urgency: DailyBrief.Urgency = daysOld > 3 ? .high : daysOld > 1 ? .medium : .low

            return DailyBrief.RelationshipNudge(
                contactName: commitment.counterparty,
                reason: commitment.commitmentDescription,
                urgency: urgency
            )
        }
    }

    private func getMeetingPreps(for date: Date) async throws -> [DailyBrief.MeetingPrep] {
        let now = Date()
        guard let threeHours = Calendar.current.date(byAdding: .hour, value: 3, to: now) else {
            return []
        }

        let predicate = eventStore.predicateForEvents(
            withStart: now, end: threeHours, calendars: nil
        )
        let upcomingEvents = eventStore.events(matching: predicate)
            .filter { !$0.isAllDay }

        var preps: [DailyBrief.MeetingPrep] = []

        let recallEngine = RecallEngine(
            memoryStore: memoryStore,
            embedder: embedder,
            embeddingIndex: embeddingIndex
        )

        for event in upcomingEvents {
            let attendeeNames = event.attendees?.compactMap { $0.name } ?? []
            let query = ([event.title ?? ""] + attendeeNames).joined(separator: " ")

            let relatedMemories = try await recallEngine.search(query: query, limit: 3)

            preps.append(DailyBrief.MeetingPrep(
                eventTitle: event.title ?? "Meeting",
                startTime: event.startDate,
                context: relatedMemories.map { $0.memory.content }.joined(separator: ". "),
                relevantMemories: relatedMemories.map { $0.memory.id }
            ))
        }

        return preps
    }

    // MARK: - Helpers

    private func calculateFreeBlocks(
        events: [EKEvent],
        dayStart: Date,
        dayEnd: Date
    ) -> [String] {
        var freeBlocks: [String] = []
        let workdayStart = Calendar.current.date(bySettingHour: 9, minute: 0, second: 0, of: dayStart)!
        let workdayEnd = Calendar.current.date(bySettingHour: 18, minute: 0, second: 0, of: dayStart)!

        var currentTime = workdayStart

        for event in events {
            guard event.startDate > currentTime else {
                currentTime = max(currentTime, event.endDate)
                continue
            }

            let gapMinutes = Int(event.startDate.timeIntervalSince(currentTime) / 60)
            if gapMinutes >= 30 {
                freeBlocks.append("\(formatTime(currentTime))-\(formatTime(event.startDate)) is open")
            }
            currentTime = event.endDate
        }

        // Check gap after last event
        if currentTime < workdayEnd {
            let gapMinutes = Int(workdayEnd.timeIntervalSince(currentTime) / 60)
            if gapMinutes >= 30 {
                freeBlocks.append("\(formatTime(currentTime))-\(formatTime(workdayEnd)) is open")
            }
        }

        return freeBlocks
    }

    private func formatTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mma"
        return formatter.string(from: date).lowercased()
    }
}
