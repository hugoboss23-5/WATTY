import Foundation
import BackgroundTasks
import UserNotifications

/// Manages all background tasks: brief generation, data ingestion, and nightly clustering.
/// Brief generates at 6am, delivers at 7am. Ingestion runs every 2-4 hours.
/// Clustering runs at 3am.
/// @unchecked Sendable: singleton with no mutable state; BGTaskScheduler closures
/// require Sendable captures under Swift 6.
final class BriefScheduler: @unchecked Sendable {
    static let shared = BriefScheduler()

    // Task identifiers — must match Info.plist BGTaskSchedulerPermittedIdentifiers
    static let briefGenerationID = "com.watty.brief.generate"
    static let ingestionID = "com.watty.data.ingest"
    static let clusteringID = "com.watty.clustering.nightly"

    func registerTasks() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.briefGenerationID,
            using: nil
        ) { task in
            self.handleBriefGeneration(task as! BGAppRefreshTask)
        }

        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.ingestionID,
            using: nil
        ) { task in
            self.handleIngestion(task as! BGAppRefreshTask)
        }

        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.clusteringID,
            using: nil
        ) { task in
            self.handleClustering(task as! BGProcessingTask)
        }

        scheduleBriefGeneration()
        scheduleIngestion()
        scheduleClustering()
    }

    // MARK: - Brief Generation

    private func scheduleBriefGeneration() {
        let request = BGAppRefreshTaskRequest(identifier: Self.briefGenerationID)
        let tomorrow6am = Calendar.current.nextDate(
            after: Date(),
            matching: DateComponents(hour: 6, minute: 0),
            matchingPolicy: .nextTime
        )!
        request.earliestBeginDate = tomorrow6am
        try? BGTaskScheduler.shared.submit(request)
    }

    private func handleBriefGeneration(_ task: BGAppRefreshTask) {
        scheduleBriefGeneration()

        let operation = Task { @MainActor in
            do {
                let generator = BriefGenerator()
                let brief = try await generator.generate(for: Date())

                // Schedule notification for 7am delivery
                let content = UNMutableNotificationContent()
                content.title = "watty"
                content.body = brief.formattedText
                content.sound = .default
                content.interruptionLevel = .timeSensitive

                var trigger = DateComponents()
                trigger.hour = 7
                trigger.minute = 0

                let notifRequest = UNNotificationRequest(
                    identifier: "daily-brief-\(Date().formatted(.iso8601))",
                    content: content,
                    trigger: UNCalendarNotificationTrigger(
                        dateMatching: trigger, repeats: false
                    )
                )

                try await UNUserNotificationCenter.current().add(notifRequest)

                task.setTaskCompleted(success: true)
            } catch {
                task.setTaskCompleted(success: false)
            }
        }

        task.expirationHandler = {
            operation.cancel()
        }
    }

    // MARK: - Data Ingestion

    private func scheduleIngestion() {
        let request = BGAppRefreshTaskRequest(identifier: Self.ingestionID)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 2 * 60 * 60) // 2 hours
        try? BGTaskScheduler.shared.submit(request)
    }

    private func handleIngestion(_ task: BGAppRefreshTask) {
        scheduleIngestion()

        let operation = Task { @MainActor in
            do {
                let fourHoursAgo = Date(timeIntervalSinceNow: -4 * 60 * 60)
                try await IngestionPipeline.shared.runIngestion(since: fourHoursAgo)
                task.setTaskCompleted(success: true)
            } catch {
                task.setTaskCompleted(success: false)
            }
        }

        task.expirationHandler = {
            operation.cancel()
        }
    }

    // MARK: - Nightly Clustering

    private func scheduleClustering() {
        let request = BGProcessingTaskRequest(identifier: Self.clusteringID)
        let tonight3am = Calendar.current.nextDate(
            after: Date(),
            matching: DateComponents(hour: 3, minute: 0),
            matchingPolicy: .nextTime
        )!
        request.earliestBeginDate = tonight3am
        request.requiresNetworkConnectivity = false
        request.requiresExternalPower = false
        try? BGTaskScheduler.shared.submit(request)
    }

    private func handleClustering(_ task: BGProcessingTask) {
        scheduleClustering()

        let operation = Task { @MainActor in
            do {
                try await ClusterEngine.shared.rebuildClusters()
                task.setTaskCompleted(success: true)
            } catch {
                task.setTaskCompleted(success: false)
            }
        }

        task.expirationHandler = {
            operation.cancel()
        }
    }
}

/// K-means clustering engine for auto-organizing memories into topics.
/// @unchecked Sendable: singleton with immutable dependencies; mutable work
/// happens only inside @MainActor rebuildClusters().
final class ClusterEngine: @unchecked Sendable {
    static let shared = ClusterEngine()

    private let memoryStore = MemoryStore.shared
    private let embedder = Embedder()

    /// Rebuild all clusters from scratch using k-means on embedding vectors.
    @MainActor
    func rebuildClusters() async throws {
        let memories = try memoryStore.fetchAllMemories()
        let memoriesWithEmbeddings = memories.filter { $0.embedding != nil }

        guard memoriesWithEmbeddings.count >= 5 else { return }

        // Determine k using heuristic: sqrt(n/2), bounded by [5, 50]
        let k = max(5, min(50, Int(sqrt(Double(memoriesWithEmbeddings.count) / 2))))

        // Simple k-means clustering
        let clusters = kMeans(
            vectors: memoriesWithEmbeddings.compactMap { $0.embedding },
            ids: memoriesWithEmbeddings.map { $0.id },
            k: k,
            maxIterations: 20
        )

        // Save clusters to store
        for (centroid, memberIDs) in clusters {
            guard memberIDs.count >= 3 else { continue }

            // Generate cluster name from sample memories
            let sampleMemories = memoriesWithEmbeddings
                .filter { memberIDs.contains($0.id) }
                .prefix(5)

            let topics = sampleMemories.map { memory in
                // Extract key phrase (first 5 words)
                let words = memory.content.split(separator: " ").prefix(5)
                return words.joined(separator: " ")
            }

            let cluster = Cluster(name: topics.first ?? "Topic")
            cluster.memoryCount = memberIDs.count
            cluster.centroid = centroid
            cluster.sampleTopics = Array(topics)

            try memoryStore.saveCluster(cluster)

            // Update memory cluster assignments
            for memory in memoriesWithEmbeddings where memberIDs.contains(memory.id) {
                memory.clusterID = cluster.id
            }
        }
    }

    // MARK: - K-Means Implementation

    private func kMeans(
        vectors: [[Float]],
        ids: [UUID],
        k: Int,
        maxIterations: Int
    ) -> [([Float], [UUID])] {
        guard !vectors.isEmpty, let dim = vectors.first?.count else { return [] }

        // Initialize centroids randomly
        var centroids: [[Float]] = []
        var usedIndices = Set<Int>()
        for _ in 0..<min(k, vectors.count) {
            var idx: Int
            repeat {
                idx = Int.random(in: 0..<vectors.count)
            } while usedIndices.contains(idx)
            usedIndices.insert(idx)
            centroids.append(vectors[idx])
        }

        var assignments = [Int](repeating: 0, count: vectors.count)

        for _ in 0..<maxIterations {
            // Assign each vector to nearest centroid
            var changed = false
            for i in 0..<vectors.count {
                var bestCluster = 0
                var bestDistance: Float = .infinity
                for c in 0..<centroids.count {
                    let dist = euclideanDistance(vectors[i], centroids[c])
                    if dist < bestDistance {
                        bestDistance = dist
                        bestCluster = c
                    }
                }
                if assignments[i] != bestCluster {
                    assignments[i] = bestCluster
                    changed = true
                }
            }

            if !changed { break }

            // Recompute centroids
            for c in 0..<centroids.count {
                let members = (0..<vectors.count).filter { assignments[$0] == c }
                guard !members.isEmpty else { continue }

                var newCentroid = [Float](repeating: 0, count: dim)
                for m in members {
                    for d in 0..<dim {
                        newCentroid[d] += vectors[m][d]
                    }
                }
                for d in 0..<dim {
                    newCentroid[d] /= Float(members.count)
                }
                centroids[c] = newCentroid
            }
        }

        // Build result
        var result: [([Float], [UUID])] = []
        for c in 0..<centroids.count {
            let memberIDs = (0..<vectors.count)
                .filter { assignments[$0] == c }
                .map { ids[$0] }
            if !memberIDs.isEmpty {
                result.append((centroids[c], memberIDs))
            }
        }
        return result
    }

    private func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return .infinity }
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
}
