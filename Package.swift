// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "MLXLoRAStudio",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "MLXLoRAStudio", targets: ["MLXLoRAStudio"])
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "MLXLoRAStudio",
            path: "Sources/MLXLoRAStudio"
        ),
        .testTarget(
            name: "MLXLoRAStudioTests",
            dependencies: ["MLXLoRAStudio"],
            path: "Tests/MLXLoRAStudioTests"
        )
    ]
)
