import Foundation
import WattyCore

/// watty_contact_context — Get relationship context for any contact.
/// When you last talked, what you discussed, open commitments, and patterns.
struct WattyContactContext: MCPTool {
    private let memoryStore = MemoryStore.shared
    private let recallEngine = RecallEngine()
    private let formatter = TextFormatter()

    var definition: [String: Any] {
        [
            "name": "watty_contact_context",
            "description": "Get context about your relationship with someone — when you last talked, what you discussed, open commitments between you, and anything notable about recent interactions.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "name": [
                        "type": "string",
                        "description": "The person's name."
                    ] as [String: Any]
                ] as [String: Any],
                "required": ["name"]
            ] as [String: Any]
        ]
    }

    @MainActor
    func execute(params: JSONRPCParams) async throws -> Any {
        guard let name = params.string(for: "name") else {
            throw MCPError.missingParameter("name")
        }

        // Fetch contact profile
        let profile = try memoryStore.fetchContactProfile(name: name)

        // Fetch commitments involving this person
        let allCommitments = try memoryStore.fetchCommitments(status: .open)
        let contactCommitments = allCommitments.filter {
            $0.counterparty.lowercased().contains(name.lowercased())
        }

        // Count memories mentioning this person
        let allMemories = try memoryStore.fetchAllMemories()
        let contactMemories = allMemories.filter {
            $0.contactNames.contains(where: { $0.lowercased().contains(name.lowercased()) })
        }

        // Build frequency description
        let frequency: String
        if let freq = profile?.interactionFrequency {
            if freq > 5 { frequency = "daily" }
            else if freq > 2 { frequency = "several times a week" }
            else if freq > 0.5 { frequency = "weekly" }
            else { frequency = "occasional" }
        } else {
            frequency = "unknown"
        }

        let now = Date()
        let commitmentSnapshots: [[String: Any]] = contactCommitments.map { c in
            let daysOld = Calendar.current.dateComponents([.day], from: c.createdAt, to: now).day ?? 0
            return [
                "description": c.commitmentDescription,
                "days_old": daysOld
            ] as [String: Any]
        }

        return [
            "contact": [
                "name": profile?.displayName ?? name,
                "last_interaction": profile?.lastInteraction.map {
                    ISO8601DateFormatter().string(from: $0)
                } as Any,
                "interaction_frequency": frequency,
                "recent_topics": profile?.recentTopics ?? [],
                "open_commitments": commitmentSnapshots,
                "memory_count": contactMemories.count
            ] as [String: Any]
        ] as [String: Any]
    }
}
