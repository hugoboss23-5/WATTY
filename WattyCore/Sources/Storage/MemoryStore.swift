import SwiftData
import Foundation

/// SwiftData persistent store for all Watty data — encrypted on-device.
final class MemoryStore: Sendable {
    static let shared = MemoryStore()

    let container: ModelContainer

    init() {
        let schema = Schema([
            Memory.self,
            Commitment.self,
            ContactProfile.self,
            Cluster.self
        ])

        let config = ModelConfiguration(
            schema: schema,
            isStoredInMemoryOnly: false,
            groupContainer: .identifier("group.com.watty.shared")
        )

        self.container = try! ModelContainer(for: schema, configurations: [config])
    }

    /// For testing — in-memory store
    init(inMemory: Bool) {
        let schema = Schema([
            Memory.self,
            Commitment.self,
            ContactProfile.self,
            Cluster.self
        ])

        let config = ModelConfiguration(
            schema: schema,
            isStoredInMemoryOnly: true
        )

        self.container = try! ModelContainer(for: schema, configurations: [config])
    }

    // MARK: - Memory CRUD

    @MainActor
    func save(_ memory: Memory) throws {
        let context = container.mainContext
        context.insert(memory)
        try context.save()
    }

    @MainActor
    func saveAll(_ memories: [Memory]) throws {
        let context = container.mainContext
        for memory in memories {
            context.insert(memory)
        }
        try context.save()
    }

    @MainActor
    func fetchMemories(
        since: Date? = nil,
        source: MemorySource? = nil,
        limit: Int = 100
    ) throws -> [Memory] {
        let context = container.mainContext
        var descriptor = FetchDescriptor<Memory>(
            sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
        )
        descriptor.fetchLimit = limit

        if let since, let source {
            descriptor.predicate = #Predicate<Memory> {
                $0.createdAt > since && $0.source == source
            }
        } else if let since {
            descriptor.predicate = #Predicate<Memory> {
                $0.createdAt > since
            }
        } else if let source {
            descriptor.predicate = #Predicate<Memory> {
                $0.source == source
            }
        }

        return try context.fetch(descriptor)
    }

    @MainActor
    func fetchAllMemories() throws -> [Memory] {
        let context = container.mainContext
        let descriptor = FetchDescriptor<Memory>(
            sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
        )
        return try context.fetch(descriptor)
    }

    // MARK: - Commitment CRUD

    @MainActor
    func saveCommitment(_ commitment: Commitment) throws {
        let context = container.mainContext
        context.insert(commitment)
        try context.save()
    }

    @MainActor
    func fetchCommitments(status: CommitmentStatus) throws -> [Commitment] {
        let context = container.mainContext
        var descriptor = FetchDescriptor<Commitment>(
            sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
        )
        descriptor.predicate = #Predicate<Commitment> {
            $0.status == status
        }
        return try context.fetch(descriptor)
    }

    @MainActor
    func fetchAllCommitments() throws -> [Commitment] {
        let context = container.mainContext
        let descriptor = FetchDescriptor<Commitment>(
            sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
        )
        return try context.fetch(descriptor)
    }

    // MARK: - ContactProfile CRUD

    @MainActor
    func saveContactProfile(_ profile: ContactProfile) throws {
        let context = container.mainContext
        context.insert(profile)
        try context.save()
    }

    @MainActor
    func fetchContactProfile(name: String) throws -> ContactProfile? {
        let context = container.mainContext
        var descriptor = FetchDescriptor<ContactProfile>()
        descriptor.predicate = #Predicate<ContactProfile> {
            $0.displayName == name
        }
        descriptor.fetchLimit = 1
        return try context.fetch(descriptor).first
    }

    @MainActor
    func fetchAllContactProfiles() throws -> [ContactProfile] {
        let context = container.mainContext
        let descriptor = FetchDescriptor<ContactProfile>(
            sortBy: [SortDescriptor(\.displayName)]
        )
        return try context.fetch(descriptor)
    }

    // MARK: - Cluster CRUD

    @MainActor
    func saveCluster(_ cluster: Cluster) throws {
        let context = container.mainContext
        context.insert(cluster)
        try context.save()
    }

    @MainActor
    func fetchAllClusters() throws -> [Cluster] {
        let context = container.mainContext
        let descriptor = FetchDescriptor<Cluster>(
            sortBy: [SortDescriptor(\.updatedAt, order: .reverse)]
        )
        return try context.fetch(descriptor)
    }

    // MARK: - Counts

    @MainActor
    func totalMemoryCount() throws -> Int {
        let context = container.mainContext
        return try context.fetchCount(FetchDescriptor<Memory>())
    }

    @MainActor
    func totalCommitmentCount(status: CommitmentStatus? = nil) throws -> Int {
        let context = container.mainContext
        if let status {
            var descriptor = FetchDescriptor<Commitment>()
            descriptor.predicate = #Predicate<Commitment> {
                $0.status == status
            }
            return try context.fetchCount(descriptor)
        }
        return try context.fetchCount(FetchDescriptor<Commitment>())
    }
}
