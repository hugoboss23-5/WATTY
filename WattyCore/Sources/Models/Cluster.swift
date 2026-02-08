import SwiftData
import Foundation

/// A semantic cluster of related memories — auto-organized by embedding similarity.
@Model
final class Cluster {
    var id: UUID
    var name: String
    var createdAt: Date
    var updatedAt: Date

    // Cluster metadata
    var memoryCount: Int
    var centroid: [Float]?  // Average embedding vector for this cluster
    var sampleTopics: [String]

    init(name: String) {
        self.id = UUID()
        self.name = name
        self.createdAt = Date()
        self.updatedAt = Date()
        self.memoryCount = 0
        self.sampleTopics = []
    }
}
