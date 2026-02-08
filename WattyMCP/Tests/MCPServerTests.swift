import Testing
import Foundation
@testable import WattyMCP
@testable import WattyCore

@Suite("MCP Server Tests")
struct MCPServerTests {

    @Test("StdioTransport parses valid JSON-RPC request")
    func testRequestParsing() throws {
        let json = """
        {"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}
        """
        let data = json.data(using: .utf8)!
        let request = try JSONDecoder().decode(JSONRPCRequest.self, from: data)

        #expect(request.jsonrpc == "2.0")
        #expect(request.method == "tools/list")
        #expect(request.id == .int(1))
    }

    @Test("JSON-RPC response serializes correctly")
    func testResponseSerialization() throws {
        let response = JSONRPCResponse.success(
            id: .int(1),
            result: ["key": "value"] as [String: String]
        )
        let data = try JSONEncoder().encode(response)
        let json = String(data: data, encoding: .utf8)!

        #expect(json.contains("jsonrpc"))
        #expect(json.contains("2.0"))
    }

    @Test("JSON-RPC error response format")
    func testErrorResponse() throws {
        let response = JSONRPCResponse.error(
            id: .string("test-id"),
            code: -32601,
            message: "Method not found"
        )
        let data = try JSONEncoder().encode(response)
        let json = String(data: data, encoding: .utf8)!

        #expect(json.contains("-32601"))
        #expect(json.contains("Method not found"))
    }

    @Test("AnyCodable encodes and decodes strings")
    func testAnyCodableString() throws {
        let original = AnyCodable("hello")
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(AnyCodable.self, from: data)
        #expect(decoded.value as? String == "hello")
    }

    @Test("AnyCodable encodes and decodes integers")
    func testAnyCodableInt() throws {
        let original = AnyCodable(42)
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(AnyCodable.self, from: data)
        #expect(decoded.value as? Int == 42)
    }

    @Test("AnyCodable encodes and decodes booleans")
    func testAnyCodableBool() throws {
        let original = AnyCodable(true)
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(AnyCodable.self, from: data)
        #expect(decoded.value as? Bool == true)
    }

    @Test("Tool definitions have correct structure")
    func testToolDefinitions() {
        let recall = WattyRecall()
        let def = recall.definition

        #expect(def["name"] as? String == "watty_recall")
        #expect(def["description"] != nil)
        #expect(def["inputSchema"] != nil)

        let schema = def["inputSchema"] as? [String: Any]
        #expect(schema?["type"] as? String == "object")

        let properties = schema?["properties"] as? [String: Any]
        #expect(properties?["query"] != nil)

        let required = schema?["required"] as? [String]
        #expect(required?.contains("query") == true)
    }

    @Test("All 7 tools have unique names")
    func testUniqueToolNames() {
        let tools: [MCPTool] = [
            WattyRecall(),
            WattyBrief(),
            WattyCommitments(),
            WattyStore(),
            WattyClusters(),
            WattyCalendarPrep(),
            WattyContactContext()
        ]

        let names = tools.map { $0.definition["name"] as! String }
        let uniqueNames = Set(names)
        #expect(names.count == 7)
        #expect(uniqueNames.count == 7)
    }
}
