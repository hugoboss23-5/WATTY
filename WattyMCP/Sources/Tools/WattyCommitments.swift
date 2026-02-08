import Foundation
import WattyCore

/// watty_commitments — List open commitments extracted from conversations.
/// Shows what you owe, what's owed to you, and what's overdue.
struct WattyCommitments: MCPTool {
    private let memoryStore = MemoryStore.shared

    var definition: [String: Any] {
        [
            "name": "watty_commitments",
            "description": "List your open commitments — promises you made or were made to you, extracted from your messages and conversations. Shows what you owe, what's owed to you, and what's overdue.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "status": [
                        "type": "string",
                        "description": "Filter by status. Default: open.",
                        "enum": ["open", "completed", "expired", "all"],
                        "default": "open"
                    ] as [String: Any],
                    "contact": [
                        "type": "string",
                        "description": "Filter by person name."
                    ] as [String: Any]
                ] as [String: Any]
            ] as [String: Any]
        ]
    }

    @MainActor
    func execute(params: JSONRPCParams) async throws -> Any {
        let statusStr = params.string(for: "status") ?? "open"
        let contactFilter = params.string(for: "contact")

        var commitments: [Commitment]

        if statusStr == "all" {
            commitments = try memoryStore.fetchAllCommitments()
        } else {
            let status = CommitmentStatus(rawValue: statusStr) ?? .open
            commitments = try memoryStore.fetchCommitments(status: status)
        }

        // Apply contact filter
        if let contactFilter {
            commitments = commitments.filter {
                $0.counterparty.lowercased().contains(contactFilter.lowercased())
            }
        }

        // Sort: overdue first, then by creation date
        let now = Date()
        commitments.sort { a, b in
            let aOverdue = a.dueDate.map { $0 < now } ?? false
            let bOverdue = b.dueDate.map { $0 < now } ?? false
            if aOverdue != bOverdue { return aOverdue }
            return a.createdAt > b.createdAt
        }

        let commitmentDicts: [[String: Any]] = commitments.map { c in
            let daysOld = Calendar.current.dateComponents([.day], from: c.createdAt, to: now).day ?? 0
            var dict: [String: Any] = [
                "id": c.id.uuidString,
                "description": c.commitmentDescription,
                "counterparty": c.counterparty,
                "you_promised": c.ownerIsUser,
                "created_at": ISO8601DateFormatter().string(from: c.createdAt),
                "status": c.status.rawValue,
                "days_old": daysOld
            ]
            if let dueDate = c.dueDate {
                dict["due_date"] = ISO8601DateFormatter().string(from: dueDate)
            }
            return dict
        }

        // Summary stats
        let allCommitments = try memoryStore.fetchAllCommitments()
        let openCount = allCommitments.filter { $0.status == .open }.count
        let overdueCount = allCommitments.filter { c in
            c.status == .open && (c.dueDate.map { $0 < now } ?? false)
        }.count
        let weekAgo = Calendar.current.date(byAdding: .weekOfYear, value: -1, to: now)!
        let completedThisWeek = allCommitments.filter { c in
            c.status == .completed && (c.completedAt.map { $0 > weekAgo } ?? false)
        }.count

        return [
            "commitments": commitmentDicts,
            "summary": [
                "open": openCount,
                "overdue": overdueCount,
                "completed_this_week": completedThisWeek
            ] as [String: Any]
        ] as [String: Any]
    }
}
