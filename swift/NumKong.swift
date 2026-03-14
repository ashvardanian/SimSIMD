//  NumKong.swift
//  NumKong
//
//  Created by Ash Vardanian on March 14, 2026.
//

import CNumKong

// MARK: - Geospatial Protocols

/// A type that can compute SIMD-accelerated Haversine (great-circle) distances.
public protocol NumKongHaversine {
    static func haversine(
        aLat: UnsafeBufferPointer<Self>,
        aLon: UnsafeBufferPointer<Self>,
        bLat: UnsafeBufferPointer<Self>,
        bLon: UnsafeBufferPointer<Self>,
        result: UnsafeMutableBufferPointer<Self>
    ) -> Bool
}

/// A type that can compute SIMD-accelerated Vincenty (ellipsoidal) geodesic distances.
public protocol NumKongVincenty {
    static func vincenty(
        aLat: UnsafeBufferPointer<Self>,
        aLon: UnsafeBufferPointer<Self>,
        bLat: UnsafeBufferPointer<Self>,
        bLon: UnsafeBufferPointer<Self>,
        result: UnsafeMutableBufferPointer<Self>
    ) -> Bool
}

/// Convenience alias for types supporting both Haversine and Vincenty geospatial distances.
public typealias NumKongGeospatial = NumKongHaversine & NumKongVincenty

extension Float64: NumKongHaversine {
    @inlinable @inline(__always)
    public static func haversine(
        aLat: UnsafeBufferPointer<Float64>,
        aLon: UnsafeBufferPointer<Float64>,
        bLat: UnsafeBufferPointer<Float64>,
        bLon: UnsafeBufferPointer<Float64>,
        result: UnsafeMutableBufferPointer<Float64>
    ) -> Bool {
        let n = aLat.count
        guard
            n > 0 && n == aLon.count && n == bLat.count && n == bLon.count
                && n == result.count
        else {
            return false
        }
        nk_haversine_f64(
            aLat.baseAddress!,
            aLon.baseAddress!,
            bLat.baseAddress!,
            bLon.baseAddress!,
            UInt64(n),
            result.baseAddress!
        )
        return true
    }
}

extension Float64: NumKongVincenty {
    @inlinable @inline(__always)
    public static func vincenty(
        aLat: UnsafeBufferPointer<Float64>,
        aLon: UnsafeBufferPointer<Float64>,
        bLat: UnsafeBufferPointer<Float64>,
        bLon: UnsafeBufferPointer<Float64>,
        result: UnsafeMutableBufferPointer<Float64>
    ) -> Bool {
        let n = aLat.count
        guard
            n > 0 && n == aLon.count && n == bLat.count && n == bLon.count
                && n == result.count
        else {
            return false
        }
        nk_vincenty_f64(
            aLat.baseAddress!,
            aLon.baseAddress!,
            bLat.baseAddress!,
            bLon.baseAddress!,
            UInt64(n),
            result.baseAddress!
        )
        return true
    }
}

extension Float32: NumKongHaversine {
    @inlinable @inline(__always)
    public static func haversine(
        aLat: UnsafeBufferPointer<Float32>,
        aLon: UnsafeBufferPointer<Float32>,
        bLat: UnsafeBufferPointer<Float32>,
        bLon: UnsafeBufferPointer<Float32>,
        result: UnsafeMutableBufferPointer<Float32>
    ) -> Bool {
        let n = aLat.count
        guard
            n > 0 && n == aLon.count && n == bLat.count && n == bLon.count
                && n == result.count
        else {
            return false
        }
        nk_haversine_f32(
            aLat.baseAddress!,
            aLon.baseAddress!,
            bLat.baseAddress!,
            bLon.baseAddress!,
            UInt64(n),
            result.baseAddress!
        )
        return true
    }
}

extension Float32: NumKongVincenty {
    @inlinable @inline(__always)
    public static func vincenty(
        aLat: UnsafeBufferPointer<Float32>,
        aLon: UnsafeBufferPointer<Float32>,
        bLat: UnsafeBufferPointer<Float32>,
        bLon: UnsafeBufferPointer<Float32>,
        result: UnsafeMutableBufferPointer<Float32>
    ) -> Bool {
        let n = aLat.count
        guard
            n > 0 && n == aLon.count && n == bLat.count && n == bLon.count
                && n == result.count
        else {
            return false
        }
        nk_vincenty_f32(
            aLat.baseAddress!,
            aLon.baseAddress!,
            bLat.baseAddress!,
            bLon.baseAddress!,
            UInt64(n),
            result.baseAddress!
        )
        return true
    }
}

// MARK: - Capabilities

/// Bitmask of SIMD instruction sets detected at runtime.
public enum Capabilities {
    public static var available: UInt64 { nk_capabilities() }

    public static let neon: UInt64 = 1 << 1
    public static let haswell: UInt64 = 1 << 2
    public static let skylake: UInt64 = 1 << 3
    public static let neonHalf: UInt64 = 1 << 4
    public static let neonSDot: UInt64 = 1 << 5
    public static let neonFhm: UInt64 = 1 << 6
    public static let icelake: UInt64 = 1 << 7
    public static let genoa: UInt64 = 1 << 8
    public static let neonBfDot: UInt64 = 1 << 9
    public static let sve: UInt64 = 1 << 10
    public static let sveHalf: UInt64 = 1 << 11
    public static let sveSDot: UInt64 = 1 << 12
    public static let alder: UInt64 = 1 << 13
    public static let sveBfDot: UInt64 = 1 << 14
    public static let sve2: UInt64 = 1 << 15
    public static let v128Relaxed: UInt64 = 1 << 16
    public static let sapphire: UInt64 = 1 << 17
    public static let sapphireAmx: UInt64 = 1 << 18
    public static let rvv: UInt64 = 1 << 19
    public static let rvvHalf: UInt64 = 1 << 20
    public static let rvvBf16: UInt64 = 1 << 21
    public static let graniteAmx: UInt64 = 1 << 22
    public static let turin: UInt64 = 1 << 23
    public static let sme: UInt64 = 1 << 24
    public static let sme2: UInt64 = 1 << 25
    public static let smeF64: UInt64 = 1 << 26
    public static let smeFa64: UInt64 = 1 << 27
    public static let sve2p1: UInt64 = 1 << 28
    public static let sme2p1: UInt64 = 1 << 29
    public static let smeHalf: UInt64 = 1 << 30
    public static let smeBf16: UInt64 = 1 << 31
    public static let smeLut2: UInt64 = 1 << 32
    public static let rvvBB: UInt64 = 1 << 33
    public static let sierra: UInt64 = 1 << 34
}
