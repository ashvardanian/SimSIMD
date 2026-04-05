// swift-tools-version:5.4
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "NumKong",
    platforms: [
        .macOS(.v11),
        .iOS(.v14),
        .tvOS(.v14),
        .watchOS(.v7),
    ],
    products: [
        // We need to expose the underlying `CNumKongDispatch` target to simplify
        // linking for USearch
        .library(name: "NumKong", targets: ["NumKong"]),
        .library(name: "CNumKongDispatch", targets: ["CNumKongDispatch"]),
    ],
    targets: [
        .testTarget(
            name: "Test",
            dependencies: ["NumKong"],
            path: "test/swift",
            cSettings: [
                .define("NK_DYNAMIC_DISPATCH", to: "1"),
                .define("NK_NATIVE_F16", to: "0"),
                .define("NK_NATIVE_BF16", to: "0"),
            ]
        ),
        .testTarget(
            name: "Bench",
            dependencies: ["NumKong", "CNumKong"],
            path: "bench/swift",
            cSettings: [
                .define("NK_DYNAMIC_DISPATCH", to: "1"),
                .define("NK_NATIVE_F16", to: "0"),
                .define("NK_NATIVE_BF16", to: "0"),
            ]
        ),
        .target(
            name: "NumKong",
            dependencies: ["CNumKong", "CNumKongDispatch"],
            path: "swift",
            exclude: ["README.md"],
            cSettings: [
                .define("NK_DYNAMIC_DISPATCH", to: "1"),
                .define("NK_NATIVE_F16", to: "0"),
                .define("NK_NATIVE_BF16", to: "0"),
            ]
        ),
        .target(
            name: "CNumKong",
            path: "include",
            publicHeadersPath: ".",
            cSettings: [
                .define("NK_DYNAMIC_DISPATCH", to: "1"),
                .define("NK_NATIVE_F16", to: "0"),
                .define("NK_NATIVE_BF16", to: "0"),
            ]
        ),
        .target(
            name: "CNumKongDispatch",
            dependencies: ["CNumKong"],
            path: "c",
            cSettings: [
                .define("NK_DYNAMIC_DISPATCH", to: "1"),
                .define("NK_NATIVE_F16", to: "0"),
                .define("NK_NATIVE_BF16", to: "0"),
                .headerSearchPath("../include"),
                .unsafeFlags(["-Wall", "-Wno-psabi"]),
            ]
        ),
    ]
)
