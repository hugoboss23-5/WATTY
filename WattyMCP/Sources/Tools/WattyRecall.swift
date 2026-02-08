import Foundation
import WattyCore

/// watty_recall — Semantic search across all memories.
/// The core product. "What was that idea about shapes and economics?"
struct WattyRecall: MCPTool {
    private let recallEngine = RecallEngine()
    private let memoryStore = MemoryStore.shared

    var definition: [String: Any] {
        [
            "name": "watty_recall",
            "description": "Search your memory semantically. Finds relevant memories by meaning, not keywords. Ask about ideas, people, projects, conversations — Watty searches across messages, notes, calendar, mail, and everything it has learned about you.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "query": [
                        "type": "string",
                        "description": "What you're looking for. Natural language. Example: 'that idea about geometric shapes and economics'"
                    ] as [String: Any],
                    "limit": [
                        "type": "integer",
                        "description": "Max results to return. Default 5.",
                        "default": 5
                    ] as [String: Any],
                    "source_filter": [
                        "type": "string",
                        "description": "Optional: filter by source.",
                        "enum": ["calendar", "message", "mail", "note", "reminder", "manual", "aiConversation"]
                    ] as [String: Any]
                ] as [String: Any],
                "required": ["query"]
            ] as [String: Any]
        ]
    }

    @MainActor
    func execute(params: JSONRPCParams) async throws -> Any {
        guard let query = params.string(for: "query") else {
            throw MCPError.missingParameter("query")
        }

        let limit = params.int(for: "limit") ?? 5
        let sourceFilter: MemorySource? = params.string(for: "source_filter")
            .flatMap { MemorySource(rawValue: $0) }

        let startTime = Date()
        let results = try await recallEngine.search(
            query: query,
            limit: limit,
            sourceFilter: sourceFilter
        )
        let queryTime = Int(Date().timeIntervalSince(startTime) * 1000)

        let totalMemories = try memoryStore.totalMemoryCount()

        let resultDicts: [[String: Any]] = results.map { result in
            [
                "memory_id": result.memory.id.uuidString,
                "content": result.memory.content,
                "source": result.memory.source.rawValue,
                "created_at": ISO8601DateFormatter().string(from: result.memory.createdAt),
                "relevance_score": result.relevanceScore,
                "contact_names": result.memory.contactNames,
                "cluster": result.clusterName ?? "uncategorized"
            ] as [String: Any]
        }

        return [
            "results": resultDicts,
            "total_memories": totalMemories,
            "query_embedding_time_ms": queryTime
        ] as [String: Any]
    }
}
