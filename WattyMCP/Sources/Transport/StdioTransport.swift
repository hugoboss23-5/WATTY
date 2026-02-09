import Foundation

/// JSON-RPC 2.0 transport over stdin/stdout per MCP specification.
/// Handles reading requests from stdin and writing responses to stdout.
/// @unchecked Sendable: FileHandle is thread-safe; instance is used only from MCPServer.main().
final class StdioTransport: @unchecked Sendable {
    private let inputHandle: FileHandle
    private let outputHandle: FileHandle

    init(
        input: FileHandle = .standardInput,
        output: FileHandle = .standardOutput
    ) {
        self.inputHandle = input
        self.outputHandle = output
    }

    /// Start the transport loop — reads JSON-RPC messages from stdin.
    func start(handler: @escaping @Sendable (JSONRPCRequest) async -> JSONRPCResponse?) async {
        while let line = readLine() {
            guard !line.isEmpty else { continue }

            guard let data = line.data(using: .utf8),
                  let request = try? JSONDecoder().decode(JSONRPCRequest.self, from: data) else {
                let error = JSONRPCResponse.error(
                    id: nil,
                    code: -32700,
                    message: "Parse error"
                )
                send(error)
                continue
            }

            if let response = await handler(request) {
                send(response)
            }
        }
    }

    /// Send a JSON-RPC response to stdout.
    func send(_ response: JSONRPCResponse) {
        guard let data = try? JSONEncoder().encode(response),
              let jsonString = String(data: data, encoding: .utf8) else {
            return
        }
        let output = jsonString + "\n"
        if let outputData = output.data(using: .utf8) {
            outputHandle.write(outputData)
        }
    }

    /// Send a JSON-RPC notification (no id, no response expected).
    func sendNotification(_ method: String, params: [String: Any]? = nil) {
        var notification: [String: Any] = [
            "jsonrpc": "2.0",
            "method": method
        ]
        if let params {
            notification["params"] = params
        }
        if let data = try? JSONSerialization.data(withJSONObject: notification),
           let jsonString = String(data: data, encoding: .utf8) {
            let output = jsonString + "\n"
            if let outputData = output.data(using: .utf8) {
                outputHandle.write(outputData)
            }
        }
    }
}

// MARK: - JSON-RPC Types

struct JSONRPCRequest: Codable, Sendable {
    let jsonrpc: String
    let id: JSONRPCId?
    let method: String
    let params: JSONRPCParams?
}

enum JSONRPCId: Codable, Equatable, Sendable {
    case string(String)
    case int(Int)

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let intVal = try? container.decode(Int.self) {
            self = .int(intVal)
        } else if let strVal = try? container.decode(String.self) {
            self = .string(strVal)
        } else {
            throw DecodingError.typeMismatch(
                JSONRPCId.self,
                DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Expected string or int")
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s): try container.encode(s)
        case .int(let i): try container.encode(i)
        }
    }
}

struct JSONRPCParams: Codable, Sendable {
    let values: [String: AnyCodable]

    init(_ dict: [String: Any]) {
        self.values = dict.mapValues { AnyCodable($0) }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let dict = try container.decode([String: AnyCodable].self)
        self.values = dict
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(values)
    }

    func string(for key: String) -> String? {
        values[key]?.value as? String
    }

    func int(for key: String) -> Int? {
        if let intVal = values[key]?.value as? Int { return intVal }
        if let doubleVal = values[key]?.value as? Double { return Int(doubleVal) }
        return nil
    }

    func bool(for key: String) -> Bool? {
        values[key]?.value as? Bool
    }

    func stringArray(for key: String) -> [String]? {
        values[key]?.value as? [String]
    }
}

struct JSONRPCResponse: Codable, Sendable {
    let jsonrpc: String
    let id: JSONRPCId?
    let result: AnyCodable?
    let error: JSONRPCError?

    static func success(id: JSONRPCId?, result: Any) -> JSONRPCResponse {
        JSONRPCResponse(jsonrpc: "2.0", id: id, result: AnyCodable(result), error: nil)
    }

    static func error(id: JSONRPCId?, code: Int, message: String) -> JSONRPCResponse {
        JSONRPCResponse(
            jsonrpc: "2.0",
            id: id,
            result: nil,
            error: JSONRPCError(code: code, message: message)
        )
    }
}

struct JSONRPCError: Codable, Sendable {
    let code: Int
    let message: String
}

/// Type-erased Codable wrapper for JSON values.
/// @unchecked Sendable: value is always a JSON primitive (Bool/Int/Double/String/Array/Dict).
struct AnyCodable: @unchecked Sendable, Codable {
    let value: Any

    init(_ value: Any) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let bool = try? container.decode(Bool.self) {
            value = bool
        } else if let int = try? container.decode(Int.self) {
            value = int
        } else if let double = try? container.decode(Double.self) {
            value = double
        } else if let string = try? container.decode(String.self) {
            value = string
        } else if let array = try? container.decode([AnyCodable].self) {
            value = array.map { $0.value }
        } else if let dict = try? container.decode([String: AnyCodable].self) {
            value = dict.mapValues { $0.value }
        } else {
            value = NSNull()
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case let bool as Bool: try container.encode(bool)
        case let int as Int: try container.encode(int)
        case let double as Double: try container.encode(double)
        case let string as String: try container.encode(string)
        case let array as [Any]:
            try container.encode(array.map { AnyCodable($0) })
        case let dict as [String: Any]:
            try container.encode(dict.mapValues { AnyCodable($0) })
        default:
            try container.encodeNil()
        }
    }
}
