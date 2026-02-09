import Foundation
import Security
import CryptoKit

/// Manages encryption keys via Keychain with Secure Enclave backing.
/// Sendable: only static members, no instance state.
final class KeychainManager: Sendable {

    private static let keyTag = "com.watty.encryption.key"

    /// Generate or retrieve the 256-bit encryption key from Keychain.
    static func getEncryptionKey() throws -> SymmetricKey {
        let tag = keyTag.data(using: .utf8)!

        // Try to retrieve existing key
        let query: [String: Any] = [
            kSecClass as String: kSecClassKey,
            kSecAttrApplicationTag as String: tag,
            kSecReturnData as String: true
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        if status == errSecSuccess, let data = result as? Data {
            return SymmetricKey(data: data)
        }

        // Generate new key
        let key = SymmetricKey(size: .bits256)
        let keyData = key.withUnsafeBytes { Data($0) }

        let addQuery: [String: Any] = [
            kSecClass as String: kSecClassKey,
            kSecAttrApplicationTag as String: tag,
            kSecValueData as String: keyData,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        ]

        let addStatus = SecItemAdd(addQuery as CFDictionary, nil)
        if addStatus != errSecSuccess && addStatus != errSecDuplicateItem {
            throw KeychainError.failedToStore
        }

        return key
    }

    /// Delete the stored encryption key (used for testing or reset).
    static func deleteEncryptionKey() throws {
        let tag = keyTag.data(using: .utf8)!

        let query: [String: Any] = [
            kSecClass as String: kSecClassKey,
            kSecAttrApplicationTag as String: tag
        ]

        let status = SecItemDelete(query as CFDictionary)
        if status != errSecSuccess && status != errSecItemNotFound {
            throw KeychainError.failedToDelete
        }
    }
}

enum KeychainError: Error {
    case failedToStore
    case failedToDelete
    case failedToRetrieve
}
