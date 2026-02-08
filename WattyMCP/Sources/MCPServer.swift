import Foundation
import WattyCore

/// MCP protocol handler — routes JSON-RPC requests to the appropriate tool.
/// Exposes exactly 7 tools per the Watty v1 specification.
@main
struct MCPServer {
    private static let transport = StdioTransport()

    // Tool handlers
    private static let recallTool = WattyRecall()
    private static let briefTool = WattyBrief()
    private static let commitmentsTool = WattyCommitments()
    private static let storeTool = WattyStore()
    private static let clustersTool = WattyClusters()
    private static let calendarPrepTool = WattyCalendarPrep()
    private static let contactContextTool = WattyContactContext()

    static func main() async {
        await transport.start { request in
            await handleRequest(request)
        }
    }

    private static func handleRequest(_ request: JSONRPCRequest) async -> JSONRPCResponse? {
        switch request.method {
        case "initialize":
            return handleInitialize(id: request.id)

        case "tools/list":
            return handleToolsList(id: request.id)

        case "tools/call":
            return await handleToolCall(id: request.id, params: request.params)

        case "notifications/initialized":
            // Client acknowledges initialization — no response needed
            return nil

        case "ping":
            return JSONRPCResponse.success(id: request.id, result: [:] as [String: String])

        default:
            return JSONRPCResponse.error(
                id: request.id,
                code: -32601,
                message: "Method not found: \(request.method)"
            )
        }
    }

    // MARK: - Initialize

    private static func handleInitialize(id: JSONRPCId?) -> JSONRPCResponse {
        let serverInfo: [String: Any] = [
            "protocolVersion": "2024-11-05",
            "capabilities": [
                "tools": [:] as [String: Any]
            ] as [String: Any],
            "serverInfo": [
                "name": "watty",
                "version": "1.0.0"
            ] as [String: Any]
        ]
        return JSONRPCResponse.success(id: id, result: serverInfo)
    }

    // MARK: - Tools List

    private static func handleToolsList(id: JSONRPCId?) -> JSONRPCResponse {
        let tools: [[String: Any]] = [
            recallTool.definition,
            briefTool.definition,
            commitmentsTool.definition,
            storeTool.definition,
            clustersTool.definition,
            calendarPrepTool.definition,
            contactContextTool.definition,
        ]
        return JSONRPCResponse.success(id: id, result: ["tools": tools])
    }

    // MARK: - Tool Call

    private static func handleToolCall(id: JSONRPCId?, params: JSONRPCParams?) async -> JSONRPCResponse {
        guard let toolName = params?.string(for: "name") else {
            return JSONRPCResponse.error(id: id, code: -32602, message: "Missing tool name")
        }

        // Extract tool arguments
        let arguments = params?.values["arguments"]?.value as? [String: Any] ?? [:]
        let toolParams = JSONRPCParams(arguments)

        do {
            let result: Any
            switch toolName {
            case "watty_recall":
                result = try await recallTool.execute(params: toolParams)
            case "watty_brief":
                result = try await briefTool.execute(params: toolParams)
            case "watty_commitments":
                result = try await commitmentsTool.execute(params: toolParams)
            case "watty_store":
                result = try await storeTool.execute(params: toolParams)
            case "watty_clusters":
                result = try await clustersTool.execute(params: toolParams)
            case "watty_calendar_prep":
                result = try await calendarPrepTool.execute(params: toolParams)
            case "watty_contact_context":
                result = try await contactContextTool.execute(params: toolParams)
            default:
                return JSONRPCResponse.error(
                    id: id, code: -32602, message: "Unknown tool: \(toolName)"
                )
            }

            // Wrap result in MCP content format
            let content: [[String: Any]] = [
                [
                    "type": "text",
                    "text": stringifyResult(result)
                ]
            ]
            return JSONRPCResponse.success(id: id, result: ["content": content])

        } catch {
            return JSONRPCResponse.error(
                id: id, code: -32000, message: "Tool error: \(error.localizedDescription)"
            )
        }
    }

    private static func stringifyResult(_ result: Any) -> String {
        if let data = try? JSONSerialization.data(
            withJSONObject: result, options: [.prettyPrinted, .sortedKeys]
        ), let string = String(data: data, encoding: .utf8) {
            return string
        }
        return String(describing: result)
    }
}

/// Protocol for MCP tool implementations.
protocol MCPTool {
    /// The tool's JSON definition for tools/list.
    var definition: [String: Any] { get }
    /// Execute the tool with given parameters.
    func execute(params: JSONRPCParams) async throws -> Any
}
