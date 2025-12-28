// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "NumKong",
    products: [
        .library(name: "NumKong", targets: ["NumKong"]),
    ],
    targets: [
        .testTarget(name: "Test", dependencies: ["NumKong"], path: "swift", exclude:["NumKong.swift"]),
        .target(name: "NumKong", dependencies: ["CNumKong"], path: "swift", exclude:["Test.swift"]),
        .target(
            name: "CNumKong",
            path: "include/numkong/", // Adjust the path to include your C source files
            sources: ["../../c/numkong.c"], // Include the source file here
            publicHeadersPath: ".",
            cSettings: [
                .define("NK_DYNAMIC_DISPATCH", to: "1"), // Define a C macro
                .define("NK_NATIVE_F16", to: "0"), // Define a C macro
                .define("NK_NATIVE_BF16", to: "0"), // Define a C macro
                .headerSearchPath("include/"), // Specify header search paths
                .unsafeFlags(["-Wall"]) // Use with caution: specify custom compiler flags
            ]
        ),
    ]
)
