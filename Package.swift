// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Watty",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(name: "WattyCore", targets: ["WattyCore"]),
        .executable(name: "WattyMCP", targets: ["WattyMCP"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "WattyCore",
            dependencies: [],
            path: "WattyCore/Sources"
        ),
        .executableTarget(
            name: "WattyMCP",
            dependencies: ["WattyCore"],
            path: "WattyMCP/Sources"
        ),
        .testTarget(
            name: "WattyCoreTests",
            dependencies: ["WattyCore"],
            path: "WattyCore/Tests"
        ),
        .testTarget(
            name: "WattyMCPTests",
            dependencies: ["WattyMCP"],
            path: "WattyMCP/Tests"
        ),
    ]
)
