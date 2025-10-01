// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "MathKong",
  products: [
    .library(name: "MathKong", targets: ["MathKong"])
  ],
  targets: [
    .testTarget(
      name: "Test", dependencies: ["MathKong"], path: "swift", exclude: ["MathKong.swift"]),
    .target(name: "MathKong", dependencies: ["CMathKong"], path: "swift", exclude: ["Test.swift"]),
    .target(
      name: "CMathKong",
      path: "include/mathkong/",  // Adjust the path to include your C source files
      sources: ["../../c/lib.c"],  // Include the source file here
      publicHeadersPath: ".",
      cSettings: [
        .define("MATHKONG_DYNAMIC_DISPATCH", to: "1"),  // Define a C macro
        .define("MATHKONG_NATIVE_F16", to: "0"),  // Define a C macro
        .define("MATHKONG_NATIVE_BF16", to: "0"),  // Define a C macro
        .headerSearchPath("include/"),  // Specify header search paths
        .unsafeFlags(["-Wall"]),  // Use with caution: specify custom compiler flags
      ]
    ),
  ]
)
