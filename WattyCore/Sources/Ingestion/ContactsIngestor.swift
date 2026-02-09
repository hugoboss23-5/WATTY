import Foundation
import Contacts

/// Ingests contact metadata from Contacts framework → ContactProfile objects.
/// Working ingestor — Contacts framework is available on iOS 9+.
/// @unchecked Sendable: CNContactStore is thread-safe for read operations.
final class ContactsIngestor: @unchecked Sendable, IngestorProtocol {
    let source: MemorySource = .contact

    private let contactStore: CNContactStore

    init(contactStore: CNContactStore = CNContactStore()) {
        self.contactStore = contactStore
    }

    func ingest(since: Date) async throws -> [Memory] {
        let keysToFetch: [CNKeyDescriptor] = [
            CNContactGivenNameKey as CNKeyDescriptor,
            CNContactFamilyNameKey as CNKeyDescriptor,
            CNContactOrganizationNameKey as CNKeyDescriptor,
            CNContactJobTitleKey as CNKeyDescriptor,
            CNContactNoteKey as CNKeyDescriptor,
            CNContactBirthdayKey as CNKeyDescriptor,
            CNContactEmailAddressesKey as CNKeyDescriptor,
            CNContactIdentifierKey as CNKeyDescriptor
        ]

        let request = CNContactFetchRequest(keysToFetch: keysToFetch)
        var memories: [Memory] = []

        try contactStore.enumerateContacts(with: request) { contact, _ in
            let name = [contact.givenName, contact.familyName]
                .filter { !$0.isEmpty }
                .joined(separator: " ")
            guard !name.isEmpty else { return }

            var content = "Contact: \(name)"
            if !contact.organizationName.isEmpty {
                content += " at \(contact.organizationName)"
            }
            if !contact.jobTitle.isEmpty {
                content += " (\(contact.jobTitle))"
            }
            if let birthday = contact.birthday?.date {
                content += ". Birthday: \(birthday.formatted(.dateTime.month().day()))"
            }
            if !contact.note.isEmpty {
                content += ". Notes: \(contact.note)"
            }

            let memory = Memory(content: content, source: .contact)
            memory.sourceID = contact.identifier
            memory.contactNames = [name]
            memories.append(memory)
        }

        return memories
    }

    func requestPermissions() async throws -> Bool {
        return try await withCheckedThrowingContinuation { continuation in
            contactStore.requestAccess(for: .contacts) { granted, error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: granted)
                }
            }
        }
    }

    var hasPermissions: Bool {
        CNContactStore.authorizationStatus(for: .contacts) == .authorized
    }
}
