// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SimSIMD",
    products: [
        .library(name: "SimSIMD", targets: ["SimSIMD"]),
    ],
    targets: [
        .testTarget(name: "SimSIMDTests", dependencies: ["SimSIMD"], path: "swift/tests"),
        .target(name: "SimSIMD", dependencies: ["CSimSIMD"], path: "swift/source"),
        .systemLibrary(name: "CSimSIMD", path: "include")
    ]
)
