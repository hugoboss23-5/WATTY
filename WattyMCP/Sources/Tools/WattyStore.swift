import Foundation
import WattyCore

/// watty_store — Explicitly save a memory from any AI conversation.
/// Use when the user discovers an insight, makes a decision, or wants to
/// remember something from the current conversation.
struct WattyStore: MCPTool {
    private let memoryStore = MemoryStore.shared
    private let similaritySearch = SimilaritySearch(
        embedder: Embedder(),
        index: EmbeddingIndex()
    )

    var definition: [String: Any] {
        [
            "name": "watty_store",
            "description": "Save something important to Watty's memory. Use this when the user discovers an insight, makes a decision, or wants to remember something from this conversation. Watty will index it semantically so it can be found later by meaning.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "content": [
                        "type": "string",
                        "description": "The memory to save. Be specific and self-contained — this should make sense when recalled later without context."
                    ] as [String: Any],
                    "contacts": [
                        "type": "array",
                        "items": ["type": "string"] as [String: Any],
                        "description": "People related to this memory."
                    ] as [String: Any]
                ] as [String: Any],
                "required": ["content"]
            ] as [String: Any]
        ]
    }

    @MainActor
    func execute(params: JSONRPCParams) async throws -> Any {
        guard let content = params.string(for: "content") else {
            throw MCPError.missingParameter("content")
        }

        let contacts = params.stringArray(for: "contacts") ?? []

        // Create memory
        let memory = Memory(content: content, source: .aiConversation)
        memory.contactNames = contacts
        memory.importance = 0.7  // Explicitly stored memories start higher

        // Save to store
        try memoryStore.save(memory)

        // Index embedding
        similaritySearch.indexMemory(memory)

        return [
            "stored": true,
            "memory_id": memory.id.uuidString,
            "content": memory.content,
            "source": memory.source.rawValue,
            "contacts": contacts,
            "message": "memory saved. you can find it later by meaning."
        ] as [String: Any]
    }
}
