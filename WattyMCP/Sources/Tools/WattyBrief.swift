import Foundation
import WattyCore

/// watty_brief — Generate or retrieve today's daily brief.
/// The daily text message that tells you everything about your day.
struct WattyBrief: MCPTool {
    private let briefGenerator = BriefGenerator()
    private let memoryStore = MemoryStore.shared

    var definition: [String: Any] {
        [
            "name": "watty_brief",
            "description": "Get your daily brief — a concise summary of your day including calendar events, open commitments, relationship nudges, and meeting prep. Generated fresh each morning at 7am, but can be regenerated on demand.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "date": [
                        "type": "string",
                        "description": "ISO date string. Default: today.",
                        "format": "date"
                    ] as [String: Any],
                    "regenerate": [
                        "type": "boolean",
                        "description": "Force regeneration even if today's brief exists. Default: false.",
                        "default": false
                    ] as [String: Any]
                ] as [String: Any]
            ] as [String: Any]
        ]
    }

    @MainActor
    func execute(params: JSONRPCParams) async throws -> Any {
        let dateStr = params.string(for: "date")
        let date: Date
        if let dateStr {
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyy-MM-dd"
            date = formatter.date(from: dateStr) ?? Date()
        } else {
            date = Date()
        }

        let brief = try await briefGenerator.generate(for: date)

        let commitmentCount = try memoryStore.totalCommitmentCount(status: .open)

        return [
            "brief": [
                "date": brief.date.formatted(.iso8601.year().month().day()),
                "formatted_text": brief.formattedText,
                "calendar_event_count": brief.calendarSummary?.eventCount ?? 0,
                "open_commitments": commitmentCount,
                "nudges": brief.relationshipNudges.count
            ] as [String: Any]
        ] as [String: Any]
    }
}
