import Foundation

/// Formats Watty's output in its distinctive voice.
/// This defines Watty's personality — direct, lowercase, like a text from a friend.
///
/// Uses Apple Foundation Models (`LanguageModelSession`) when available.
/// Falls back to template-based formatting for pre-Xcode 26 development.
final class TextFormatter {

    /// Watty's system prompt — the soul of the product.
    static let systemPrompt = """
    You are Watty. You text like a close friend who never forgets anything.

    RULES — THESE ARE ABSOLUTE:
    - Lowercase always. No capitals unless emphasizing ONE word.
    - No greetings beyond "morning." or "hey."
    - No sign-offs. Ever.
    - No emojis. Ever.
    - Max 3 sentences per topic. Usually 1-2.
    - Never explain how you know something. Just say it.
    - Never ask "would you like me to..." — just do it or suggest it.
    - Separate topics with a blank line.
    - Total message: under 150 words. Aim for 80.
    - Sound like someone who's known the user for years.
    - Be direct about bad news. Don't soften it.
    - If there's nothing important, just say "clear day. go create something."
    """

    /// Format a complete daily brief as a text message.
    /// Uses template-based formatting (Foundation Models integration via @Generable
    /// would replace this when running on Xcode 26+).
    func formatBrief(
        calendar: DailyBrief.CalendarSection?,
        commitments: [DailyBrief.CommitmentSnapshot],
        nudges: [DailyBrief.RelationshipNudge],
        preps: [DailyBrief.MeetingPrep],
        date: Date
    ) async throws -> String {
        var sections: [String] = []

        // Opening
        sections.append("morning.")

        // Calendar
        if let cal = calendar {
            var calText = ""
            if cal.eventCount == 1 {
                calText = cal.firstEvent.lowercased() + "."
            } else {
                calText = "\(cal.eventCount) things today. \(cal.firstEvent.lowercased()) is first."
            }
            if !cal.freeBlocks.isEmpty {
                calText += " \(cal.freeBlocks.first!.lowercased())."
            }
            sections.append(calText)
        } else {
            sections.append("nothing on the calendar.")
        }

        // Commitments
        let userCommitments = commitments.filter { $0.ownerIsUser }
        if !userCommitments.isEmpty {
            var commitmentText = ""
            for c in userCommitments.prefix(3) {
                let desc = c.description.lowercased()
                if c.daysOld > 3 {
                    commitmentText += "\(c.counterparty.lowercased()) is still waiting on you — \(desc). \(c.daysOld) days now.\n\n"
                } else if c.daysOld > 1 {
                    commitmentText += "\(c.counterparty.lowercased()) — you said you'd \(desc).\n\n"
                } else {
                    commitmentText += "don't forget: \(desc) for \(c.counterparty.lowercased()).\n\n"
                }
            }
            sections.append(commitmentText.trimmingCharacters(in: .whitespacesAndNewlines))
        }

        // Relationship nudges
        let highUrgency = nudges.filter { $0.urgency == .high }
        for nudge in highUrgency.prefix(2) {
            sections.append("you told \(nudge.contactName.lowercased()) you'd \(nudge.reason.lowercased()). that was a while ago.")
        }

        let mediumUrgency = nudges.filter { $0.urgency == .medium }
        for nudge in mediumUrgency.prefix(1) {
            sections.append("\(nudge.reason.lowercased()) — \(nudge.contactName.lowercased()).")
        }

        // Meeting prep
        for prep in preps.prefix(2) {
            let minutesStr = Int(prep.startTime.timeIntervalSinceNow / 60)
            var prepText = "\(prep.eventTitle.lowercased()) in \(minutesStr) min."
            if !prep.context.isEmpty {
                let shortContext = String(prep.context.prefix(100)).lowercased()
                prepText += " context: \(shortContext)."
            }
            sections.append(prepText)
        }

        // If truly nothing, give the clear day message
        if calendar == nil && commitments.isEmpty && nudges.isEmpty && preps.isEmpty {
            return "morning.\n\nclear day. go create something."
        }

        return sections.joined(separator: "\n\n")
    }

    /// Format a single recall result for Siri or MCP output.
    func formatRecallResult(_ result: RecallResult) -> String {
        let source = result.memory.source.rawValue
        let date = result.memory.createdAt.formatted(.dateTime.month().day())
        return "\(result.memory.content) (\(source), \(date))"
    }

    /// Format meeting prep as a short text.
    func formatMeetingPrep(
        eventTitle: String,
        minutesUntil: Int,
        attendeeContext: [(name: String, lastTopics: [String])],
        relevantNotes: [String],
        talkingPoints: [String]
    ) -> String {
        var text = "\(eventTitle.lowercased()) in \(minutesUntil) min."

        for attendee in attendeeContext.prefix(2) {
            if !attendee.lastTopics.isEmpty {
                text += " last talked to \(attendee.name.lowercased()) about \(attendee.lastTopics.first!.lowercased())."
            }
        }

        for note in relevantNotes.prefix(2) {
            text += " \(note.lowercased())."
        }

        if !talkingPoints.isEmpty {
            text += "\n\n"
            for point in talkingPoints.prefix(3) {
                text += "- \(point.lowercased())\n"
            }
        }

        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Format contact context as a short summary.
    func formatContactContext(
        name: String,
        lastInteraction: Date?,
        recentTopics: [String],
        openCommitments: [(description: String, daysOld: Int)],
        memoryCount: Int
    ) -> String {
        var text = name.lowercased()

        if let lastInteraction {
            let days = Calendar.current.dateComponents([.day], from: lastInteraction, to: Date()).day ?? 0
            if days == 0 {
                text += " — talked today."
            } else if days == 1 {
                text += " — talked yesterday."
            } else {
                text += " — last talked \(days) days ago."
            }
        }

        if !recentTopics.isEmpty {
            text += " recent topics: \(recentTopics.prefix(3).joined(separator: ", ").lowercased())."
        }

        for commitment in openCommitments.prefix(2) {
            text += "\n\(commitment.description.lowercased()) (\(commitment.daysOld)d ago)."
        }

        text += "\n\(memoryCount) memories."

        return text
    }
}
