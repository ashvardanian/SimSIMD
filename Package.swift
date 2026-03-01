// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "NumKong",
    products: [
        .library(name: "NumKong", targets: ["NumKong"])
    ],
    targets: [
        .testTarget(
            name: "Test",
            dependencies: ["NumKong"],
            path: "swift",
            exclude: ["NumKong.swift"]
        ),
        .target(
            name: "NumKong",
            dependencies: ["CNumKong"],
            path: "swift",
            exclude: ["Test.swift"]
        ),
        .target(
            name: "CNumKong",
            path: "include",  // Path containing module.modulemap
            sources: [
                "../c/numkong.c",
                // Complex float dispatch files
                "../c/dispatch_f64c.c",
                "../c/dispatch_f32c.c",
                "../c/dispatch_bf16c.c",
                "../c/dispatch_f16c.c",
                // Real float dispatch files
                "../c/dispatch_f64.c",
                "../c/dispatch_f32.c",
                "../c/dispatch_bf16.c",
                "../c/dispatch_f16.c",
                // Exotic float dispatch files
                "../c/dispatch_e5m2.c",
                "../c/dispatch_e4m3.c",
                "../c/dispatch_e3m2.c",
                "../c/dispatch_e2m3.c",
                // Signed integer dispatch files
                "../c/dispatch_i64.c",
                "../c/dispatch_i32.c",
                "../c/dispatch_i16.c",
                "../c/dispatch_i8.c",
                "../c/dispatch_i4.c",
                // Unsigned integer dispatch files
                "../c/dispatch_u64.c",
                "../c/dispatch_u32.c",
                "../c/dispatch_u16.c",
                "../c/dispatch_u8.c",
                "../c/dispatch_u4.c",
                "../c/dispatch_u1.c",
                // Special dispatch files
                "../c/dispatch_other.c",
            ],
            publicHeadersPath: ".",
            cSettings: [
                .define("NK_DYNAMIC_DISPATCH", to: "1"),  // Define a C macro
                .define("NK_NATIVE_F16", to: "0"),  // Define a C macro
                .define("NK_NATIVE_BF16", to: "0"),  // Define a C macro
                .unsafeFlags(["-Wall", "-Wno-psabi"]),  // Use with caution: specify custom compiler flags
            ]
        ),
    ]
)
