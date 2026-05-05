// SwiftPM workaround for header-only CNumKong target.
//
// Upstream NumKong's Package.swift declares CNumKong as header-only:
//   .target(name: "CNumKong", path: "include", publicHeadersPath: ".")
// SwiftPM/Xcode 16+ refuses to build header-only C targets when they are
// transitive dependencies of an app target — the link step expects a
// CNumKong.o output that is never produced.
//
// See https://github.com/swiftlang/swift-package-manager/issues/5706
//
// This file is a no-op compile unit added solely so the target produces
// at least one .o for the relocatable link to consume. It does not affect
// runtime behaviour.
static int _cnumkong_swiftpm_dummy __attribute__((unused));
