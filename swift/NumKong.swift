import CNumKong

// MARK: - Protocols with Associated Types
//
// Each protocol uses a distinct associated type name to support types like Int8
// that have different output types for different operations (e.g., Int32 for dot,
// Float32 for angular, UInt32 for squared Euclidean).

/// Protocol for types supporting dot product computation.
public protocol NumKongDot {
    associatedtype DotOutput
    static func dot<A, B>(_ a: A, _ b: B) -> DotOutput?
    where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self
}

/// Protocol for types supporting angular (cosine) distance computation.
public protocol NumKongAngular {
    associatedtype AngularOutput
    static func angular<A, B>(_ a: A, _ b: B) -> AngularOutput?
    where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self
}

/// Protocol for types supporting squared Euclidean distance computation.
public protocol NumKongL2sq {
    associatedtype L2sqOutput
    static func sqeuclidean<A, B>(_ a: A, _ b: B) -> L2sqOutput?
    where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self
}

/// Protocol for types supporting Euclidean distance computation.
public protocol NumKongL2 {
    associatedtype L2Output
    static func euclidean<A, B>(_ a: A, _ b: B) -> L2Output?
    where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self
}

/// Combined protocol for spatial similarity operations.
public typealias NumKongSpatial = NumKongDot & NumKongAngular & NumKongL2
    & NumKongL2sq

// MARK: - Float64 Implementation

extension Float64: NumKongDot {
    public typealias DotOutput = Float64

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float64, B.Element == Float64 {
        var result: Float64 = 0
        var valid = false
        a.withContiguousStorageIfAvailable { aPtr in
            b.withContiguousStorageIfAvailable { bPtr in
                guard aPtr.count > 0 && aPtr.count == bPtr.count else { return }
                nk_dot_f64(
                    aPtr.baseAddress!,
                    bPtr.baseAddress!,
                    UInt64(aPtr.count),
                    &result
                )
                valid = true
            }
        }
        return valid ? result : nil
    }
}

extension Float64: NumKongAngular {
    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float64, B.Element == Float64 {
        var result: Float64 = 0
        var valid = false
        a.withContiguousStorageIfAvailable { aPtr in
            b.withContiguousStorageIfAvailable { bPtr in
                guard aPtr.count > 0 && aPtr.count == bPtr.count else { return }
                nk_angular_f64(
                    aPtr.baseAddress!,
                    bPtr.baseAddress!,
                    UInt64(aPtr.count),
                    &result
                )
                valid = true
            }
        }
        return valid ? result : nil
    }
}

extension Float64: NumKongL2 {
    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float64, B.Element == Float64 {
        var result: Float64 = 0
        var valid = false
        a.withContiguousStorageIfAvailable { aPtr in
            b.withContiguousStorageIfAvailable { bPtr in
                guard aPtr.count > 0 && aPtr.count == bPtr.count else { return }
                nk_euclidean_f64(
                    aPtr.baseAddress!,
                    bPtr.baseAddress!,
                    UInt64(aPtr.count),
                    &result
                )
                valid = true
            }
        }
        return valid ? result : nil
    }
}

extension Float64: NumKongL2sq {
    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float64, B.Element == Float64 {
        var result: Float64 = 0
        var valid = false
        a.withContiguousStorageIfAvailable { aPtr in
            b.withContiguousStorageIfAvailable { bPtr in
                guard aPtr.count > 0 && aPtr.count == bPtr.count else { return }
                nk_sqeuclidean_f64(
                    aPtr.baseAddress!,
                    bPtr.baseAddress!,
                    UInt64(aPtr.count),
                    &result
                )
                valid = true
            }
        }
        return valid ? result : nil
    }
}

// MARK: - Float32 Implementation

extension Float32: NumKongDot {
    public typealias DotOutput = Float32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == Float32, B.Element == Float32 {
        var result: Float32 = 0
        var valid = false
        a.withContiguousStorageIfAvailable { aPtr in
            b.withContiguousStorageIfAvailable { bPtr in
                guard aPtr.count > 0 && aPtr.count == bPtr.count else { return }
                nk_dot_f32(
                    aPtr.baseAddress!,
                    bPtr.baseAddress!,
                    UInt64(aPtr.count),
                    &result
                )
                valid = true
            }
        }
        return valid ? result : nil
    }
}

extension Float32: NumKongAngular {
    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == Float32, B.Element == Float32 {
        var result: Float32 = 0
        var valid = false
        a.withContiguousStorageIfAvailable { aPtr in
            b.withContiguousStorageIfAvailable { bPtr in
                guard aPtr.count > 0 && aPtr.count == bPtr.count else { return }
                nk_angular_f32(
                    aPtr.baseAddress!,
                    bPtr.baseAddress!,
                    UInt64(aPtr.count),
                    &result
                )
                valid = true
            }
        }
        return valid ? result : nil
    }
}

extension Float32: NumKongL2 {
    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == Float32, B.Element == Float32 {
        var result: Float32 = 0
        var valid = false
        a.withContiguousStorageIfAvailable { aPtr in
            b.withContiguousStorageIfAvailable { bPtr in
                guard aPtr.count > 0 && aPtr.count == bPtr.count else { return }
                nk_euclidean_f32(
                    aPtr.baseAddress!,
                    bPtr.baseAddress!,
                    UInt64(aPtr.count),
                    &result
                )
                valid = true
            }
        }
        return valid ? result : nil
    }
}

extension Float32: NumKongL2sq {
    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == Float32, B.Element == Float32 {
        var result: Float32 = 0
        var valid = false
        a.withContiguousStorageIfAvailable { aPtr in
            b.withContiguousStorageIfAvailable { bPtr in
                guard aPtr.count > 0 && aPtr.count == bPtr.count else { return }
                nk_sqeuclidean_f32(
                    aPtr.baseAddress!,
                    bPtr.baseAddress!,
                    UInt64(aPtr.count),
                    &result
                )
                valid = true
            }
        }
        return valid ? result : nil
    }
}

// MARK: - Float16 Implementation

#if !arch(x86_64)
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    extension Float16: NumKongDot {
        public typealias DotOutput = Float32

        @inlinable @inline(__always)
        public static func dot<A, B>(_ a: A, _ b: B) -> Float32?
        where
            A: Sequence, B: Sequence, A.Element == Float16, B.Element == Float16
        {
            var result: Float32 = 0
            var valid = false
            a.withContiguousStorageIfAvailable { aPtr in
                b.withContiguousStorageIfAvailable { bPtr in
                    guard aPtr.count > 0 && aPtr.count == bPtr.count else {
                        return
                    }
                    nk_dot_f16(
                        aPtr.baseAddress!,
                        bPtr.baseAddress!,
                        UInt64(aPtr.count),
                        &result
                    )
                    valid = true
                }
            }
            return valid ? result : nil
        }
    }

    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    extension Float16: NumKongAngular {
        @inlinable @inline(__always)
        public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
        where
            A: Sequence, B: Sequence, A.Element == Float16, B.Element == Float16
        {
            var result: Float32 = 0
            var valid = false
            a.withContiguousStorageIfAvailable { aPtr in
                b.withContiguousStorageIfAvailable { bPtr in
                    guard aPtr.count > 0 && aPtr.count == bPtr.count else {
                        return
                    }
                    nk_angular_f16(
                        aPtr.baseAddress!,
                        bPtr.baseAddress!,
                        UInt64(aPtr.count),
                        &result
                    )
                    valid = true
                }
            }
            return valid ? result : nil
        }
    }

    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    extension Float16: NumKongL2 {
        @inlinable @inline(__always)
        public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
        where
            A: Sequence, B: Sequence, A.Element == Float16, B.Element == Float16
        {
            var result: Float32 = 0
            var valid = false
            a.withContiguousStorageIfAvailable { aPtr in
                b.withContiguousStorageIfAvailable { bPtr in
                    guard aPtr.count > 0 && aPtr.count == bPtr.count else {
                        return
                    }
                    nk_euclidean_f16(
                        aPtr.baseAddress!,
                        bPtr.baseAddress!,
                        UInt64(aPtr.count),
                        &result
                    )
                    valid = true
                }
            }
            return valid ? result : nil
        }
    }

    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    extension Float16: NumKongL2sq {
        @inlinable @inline(__always)
        public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float32?
        where
            A: Sequence, B: Sequence, A.Element == Float16, B.Element == Float16
        {
            var result: Float32 = 0
            var valid = false
            a.withContiguousStorageIfAvailable { aPtr in
                b.withContiguousStorageIfAvailable { bPtr in
                    guard aPtr.count > 0 && aPtr.count == bPtr.count else {
                        return
                    }
                    nk_sqeuclidean_f16(
                        aPtr.baseAddress!,
                        bPtr.baseAddress!,
                        UInt64(aPtr.count),
                        &result
                    )
                    valid = true
                }
            }
            return valid ? result : nil
        }
    }
#endif

// MARK: - Int8 Implementation

extension Int8: NumKongDot {
    public typealias DotOutput = Int32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Int32?
    where A: Sequence, B: Sequence, A.Element == Int8, B.Element == Int8 {
        var result: Int32 = 0
        var valid = false
        a.withContiguousStorageIfAvailable { aPtr in
            b.withContiguousStorageIfAvailable { bPtr in
                guard aPtr.count > 0 && aPtr.count == bPtr.count else { return }
                nk_dot_i8(
                    aPtr.baseAddress!,
                    bPtr.baseAddress!,
                    UInt64(aPtr.count),
                    &result
                )
                valid = true
            }
        }
        return valid ? result : nil
    }
}

extension Int8: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == Int8, B.Element == Int8 {
        var result: Float32 = 0
        var valid = false
        a.withContiguousStorageIfAvailable { aPtr in
            b.withContiguousStorageIfAvailable { bPtr in
                guard aPtr.count > 0 && aPtr.count == bPtr.count else { return }
                nk_angular_i8(
                    aPtr.baseAddress!,
                    bPtr.baseAddress!,
                    UInt64(aPtr.count),
                    &result
                )
                valid = true
            }
        }
        return valid ? result : nil
    }
}

extension Int8: NumKongL2 {
    public typealias L2Output = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == Int8, B.Element == Int8 {
        var result: Float32 = 0
        var valid = false
        a.withContiguousStorageIfAvailable { aPtr in
            b.withContiguousStorageIfAvailable { bPtr in
                guard aPtr.count > 0 && aPtr.count == bPtr.count else { return }
                nk_euclidean_i8(
                    aPtr.baseAddress!,
                    bPtr.baseAddress!,
                    UInt64(aPtr.count),
                    &result
                )
                valid = true
            }
        }
        return valid ? result : nil
    }
}

extension Int8: NumKongL2sq {
    public typealias L2sqOutput = UInt32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> UInt32?
    where A: Sequence, B: Sequence, A.Element == Int8, B.Element == Int8 {
        var result: UInt32 = 0
        var valid = false
        a.withContiguousStorageIfAvailable { aPtr in
            b.withContiguousStorageIfAvailable { bPtr in
                guard aPtr.count > 0 && aPtr.count == bPtr.count else { return }
                nk_sqeuclidean_i8(
                    aPtr.baseAddress!,
                    bPtr.baseAddress!,
                    UInt64(aPtr.count),
                    &result
                )
                valid = true
            }
        }
        return valid ? result : nil
    }
}

// MARK: - Collection Extensions

extension RandomAccessCollection where Element: NumKongDot {
    @inlinable @inline(__always)
    public func dot<B>(_ b: B) -> Element.DotOutput?
    where B: Sequence, B.Element == Element {
        Element.dot(self, b)
    }
}

extension RandomAccessCollection where Element: NumKongAngular {
    @inlinable @inline(__always)
    public func angular<B>(_ b: B) -> Element.AngularOutput?
    where B: Sequence, B.Element == Element {
        Element.angular(self, b)
    }
}

extension RandomAccessCollection where Element: NumKongL2 {
    @inlinable @inline(__always)
    public func euclidean<B>(_ b: B) -> Element.L2Output?
    where B: Sequence, B.Element == Element {
        Element.euclidean(self, b)
    }
}

extension RandomAccessCollection where Element: NumKongL2sq {
    @inlinable @inline(__always)
    public func sqeuclidean<B>(_ b: B) -> Element.L2sqOutput?
    where B: Sequence, B.Element == Element {
        Element.sqeuclidean(self, b)
    }
}

// MARK: - Geospatial Protocols

/// Protocol for Haversine (spherical) great-circle distance computation.
/// Coordinates must be in radians. Output is in meters.
public protocol NumKongHaversine {
    static func haversine(
        aLat: UnsafeBufferPointer<Self>,
        aLon: UnsafeBufferPointer<Self>,
        bLat: UnsafeBufferPointer<Self>,
        bLon: UnsafeBufferPointer<Self>,
        result: UnsafeMutableBufferPointer<Self>
    ) -> Bool
}

/// Protocol for Vincenty (ellipsoidal) geodesic distance computation.
/// Coordinates must be in radians. Output is in meters.
public protocol NumKongVincenty {
    static func vincenty(
        aLat: UnsafeBufferPointer<Self>,
        aLon: UnsafeBufferPointer<Self>,
        bLat: UnsafeBufferPointer<Self>,
        bLon: UnsafeBufferPointer<Self>,
        result: UnsafeMutableBufferPointer<Self>
    ) -> Bool
}

/// Combined protocol for geospatial distance computations.
public typealias NumKongGeospatial = NumKongHaversine & NumKongVincenty

// MARK: - Float64 Geospatial

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

// MARK: - Float32 Geospatial

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

/// Check available SIMD capabilities on the current CPU.
public enum Capabilities {
    public static var available: UInt64 { nk_capabilities() }

    // Chronologically-ordered capability bit positions
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
    public static let sierra: UInt64 = 1 << 13
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
}
