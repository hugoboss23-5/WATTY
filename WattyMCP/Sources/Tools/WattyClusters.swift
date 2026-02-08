import Foundation
import WattyCore

/// watty_clusters — View auto-organized knowledge topics.
/// Clusters are created automatically from patterns in your memories.
struct WattyClusters: MCPTool {
    private let memoryStore = MemoryStore.shared

    var definition: [String: Any] {
        [
            "name": "watty_clusters",
            "description": "See how Watty has organized your knowledge into topics. Clusters are created automatically from patterns in your memories — you never configure them. Useful for seeing what you've been thinking about.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "limit": [
                        "type": "integer",
                        "description": "Max clusters to return. Default: 10.",
                        "default": 10
                    ] as [String: Any]
                ] as [String: Any]
            ] as [String: Any]
        ]
    }

    @MainActor
    func execute(params: JSONRPCParams) async throws -> Any {
        let limit = params.int(for: "limit") ?? 10

        let clusters = try memoryStore.fetchAllClusters()
        let limited = Array(clusters.prefix(limit))

        let clusterDicts: [[String: Any]] = limited.map { cluster in
            [
                "id": cluster.id.uuidString,
                "name": cluster.name,
                "memory_count": cluster.memoryCount,
                "latest_memory": ISO8601DateFormatter().string(from: cluster.updatedAt),
                "sample_topics": cluster.sampleTopics
            ] as [String: Any]
        }

        return [
            "clusters": clusterDicts
        ] as [String: Any]
    }
}
