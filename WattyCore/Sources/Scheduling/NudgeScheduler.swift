import Foundation
import UserNotifications

/// Intra-day commitment and calendar reminders.
/// Sends targeted nudges about overdue commitments and upcoming meetings.
/// Respects quiet hours (10pm–7am) and max 4 nudges per day.
/// @unchecked Sendable: singleton with immutable config; mutable access via @MainActor.
final class NudgeScheduler: @unchecked Sendable {
    static let shared = NudgeScheduler()

    private let memoryStore = MemoryStore.shared
    private let maxNudgesPerDay = 4
    private let quietHoursStart = 22  // 10pm
    private let quietHoursEnd = 7     // 7am
    private let meetingPrepMinutes = 15

    /// Check if we should send any nudges right now.
    @MainActor
    func checkAndSendNudges() async throws {
        let now = Date()
        let hour = Calendar.current.component(.hour, from: now)

        // Respect quiet hours
        guard hour >= quietHoursEnd && hour < quietHoursStart else { return }

        // Check today's nudge count
        let todayNudgeCount = await getTodayNudgeCount()
        guard todayNudgeCount < maxNudgesPerDay else { return }

        var nudgesSent = 0

        // 1. Overdue commitments
        let overdueCommitments = try getOverdueCommitments()
        for commitment in overdueCommitments.prefix(2) where nudgesSent < maxNudgesPerDay - todayNudgeCount {
            let daysOld = Calendar.current.dateComponents(
                [.day], from: commitment.createdAt, to: now
            ).day ?? 0

            let body = "\(commitment.counterparty.lowercased()) is still waiting — \(commitment.commitmentDescription.lowercased()). \(daysOld) days now."

            try await sendNudge(
                identifier: "commitment-\(commitment.id.uuidString)",
                body: body
            )

            commitment.lastRemindedAt = now
            nudgesSent += 1
        }

        // 2. Upcoming meeting prep (15 min before)
        // This is handled by the BriefScheduler's meeting prep section
    }

    // MARK: - Private

    @MainActor
    private func getOverdueCommitments() throws -> [Commitment] {
        let now = Date()
        let commitments = try memoryStore.fetchCommitments(status: .open)
        return commitments.filter { commitment in
            // Consider overdue if:
            // - Has a due date that's past, OR
            // - Is more than 3 days old with no due date
            if let dueDate = commitment.dueDate {
                return dueDate < now
            }
            let daysOld = Calendar.current.dateComponents(
                [.day], from: commitment.createdAt, to: now
            ).day ?? 0
            return daysOld > 3
        }.filter { commitment in
            // Don't re-remind within 24 hours
            if let lastReminded = commitment.lastRemindedAt {
                return now.timeIntervalSince(lastReminded) > 24 * 60 * 60
            }
            return true
        }
    }

    private func sendNudge(identifier: String, body: String) async throws {
        let content = UNMutableNotificationContent()
        content.title = "watty"
        content.body = body
        content.sound = .default
        content.interruptionLevel = .active

        let request = UNNotificationRequest(
            identifier: identifier,
            content: content,
            trigger: nil  // Deliver immediately
        )

        try await UNUserNotificationCenter.current().add(request)
    }

    private func getTodayNudgeCount() async -> Int {
        let center = UNUserNotificationCenter.current()
        let delivered = await center.deliveredNotifications()
        let today = Calendar.current.startOfDay(for: Date())
        return delivered.filter { notification in
            guard let deliveryDate = notification.date as Date? else { return false }
            return deliveryDate >= today &&
                   notification.request.identifier.starts(with: "commitment-")
        }.count
    }
}
