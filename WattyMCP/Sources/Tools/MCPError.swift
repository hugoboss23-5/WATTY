import Foundation

/// Errors specific to MCP tool execution.
enum MCPError: Error, LocalizedError {
    case missingParameter(String)
    case invalidParameter(String, String)
    case toolNotFound(String)
    case executionFailed(String)

    var errorDescription: String? {
        switch self {
        case .missingParameter(let name):
            return "Missing required parameter: \(name)"
        case .invalidParameter(let name, let reason):
            return "Invalid parameter '\(name)': \(reason)"
        case .toolNotFound(let name):
            return "Unknown tool: \(name)"
        case .executionFailed(let reason):
            return "Tool execution failed: \(reason)"
        }
    }
}
