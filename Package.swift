// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SimSIMD",
    products: [
        .library(name: "SimSIMD", targets: ["SimSIMD"]),
    ],
    targets: [
        .testTarget(name: "Test", dependencies: ["SimSIMD"], path: "swift", exclude:["SimSIMD.swift"]),
        .target(name: "SimSIMD", dependencies: ["CSimSIMD"], path: "swift", exclude:["Test.swift"]),
        .target(
            name: "CSimSIMD",
            path: "include/simsimd/", // Adjust the path to include your C source files
            sources: ["../../c/lib.c"], // Include the source file here
            publicHeadersPath: ".",
            cSettings: [
                .headerSearchPath("include/"), // Specify header search paths
                .unsafeFlags(["-Wall"]) // Use with caution: specify custom compiler flags
            ]
        ),
    ]
)
