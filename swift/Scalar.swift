import CNumKong

// MARK: - Spatial Protocols

public protocol NumKongDot {
    associatedtype DotOutput
    static func dot<A, B>(_ a: A, _ b: B) -> DotOutput?
    where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self
}

public protocol NumKongAngular {
    associatedtype AngularOutput
    static func angular<A, B>(_ a: A, _ b: B) -> AngularOutput?
    where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self
}

public protocol NumKongSqeuclidean {
    associatedtype L2sqOutput
    static func sqeuclidean<A, B>(_ a: A, _ b: B) -> L2sqOutput?
    where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self
}

public protocol NumKongEuclidean {
    associatedtype L2Output
    static func euclidean<A, B>(_ a: A, _ b: B) -> L2Output?
    where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self
}

public typealias NumKongSpatial = NumKongDot & NumKongAngular & NumKongEuclidean & NumKongSqeuclidean

// MARK: - Built-in Scalars

extension Float64: NumKongDot {
    public typealias DotOutput = Float64

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float64, B.Element == Float64 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float64 = 0
            nk_dot_f64(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Float64: NumKongAngular {
    public typealias AngularOutput = Float64

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float64, B.Element == Float64 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float64 = 0
            nk_angular_f64(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Float64: NumKongEuclidean {
    public typealias L2Output = Float64

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float64, B.Element == Float64 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float64 = 0
            nk_euclidean_f64(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Float64: NumKongSqeuclidean {
    public typealias L2sqOutput = Float64

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float64, B.Element == Float64 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float64 = 0
            nk_sqeuclidean_f64(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Float32: NumKongDot {
    public typealias DotOutput = Float64

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float32, B.Element == Float32 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float64 = 0
            nk_dot_f32(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Float32: NumKongAngular {
    public typealias AngularOutput = Float64

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float32, B.Element == Float32 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float64 = 0
            nk_angular_f32(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Float32: NumKongEuclidean {
    public typealias L2Output = Float64

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float32, B.Element == Float32 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float64 = 0
            nk_euclidean_f32(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Float32: NumKongSqeuclidean {
    public typealias L2sqOutput = Float64

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float32, B.Element == Float32 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float64 = 0
            nk_sqeuclidean_f32(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

#if !arch(x86_64)
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    extension Float16: NumKongDot {
        public typealias DotOutput = Float32

        @inlinable @inline(__always)
        public static func dot<A, B>(_ a: A, _ b: B) -> Float32?
        where A: Sequence, B: Sequence, A.Element == Float16, B.Element == Float16 {
            _nkWithDensePairMapped(a, b, map: { nk_f16_t($0.bitPattern) }) { ap, bp, n in
                var result: Float32 = 0
                nk_dot_f16(ap, bp, UInt64(n), &result)
                return result
            }
        }
    }

    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    extension Float16: NumKongAngular {
        public typealias AngularOutput = Float32

        @inlinable @inline(__always)
        public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
        where A: Sequence, B: Sequence, A.Element == Float16, B.Element == Float16 {
            _nkWithDensePairMapped(a, b, map: { nk_f16_t($0.bitPattern) }) { ap, bp, n in
                var result: Float32 = 0
                nk_angular_f16(ap, bp, UInt64(n), &result)
                return result
            }
        }
    }

    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    extension Float16: NumKongEuclidean {
        public typealias L2Output = Float32

        @inlinable @inline(__always)
        public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
        where A: Sequence, B: Sequence, A.Element == Float16, B.Element == Float16 {
            _nkWithDensePairMapped(a, b, map: { nk_f16_t($0.bitPattern) }) { ap, bp, n in
                var result: Float32 = 0
                nk_euclidean_f16(ap, bp, UInt64(n), &result)
                return result
            }
        }
    }

    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    extension Float16: NumKongSqeuclidean {
        public typealias L2sqOutput = Float32

        @inlinable @inline(__always)
        public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float32?
        where A: Sequence, B: Sequence, A.Element == Float16, B.Element == Float16 {
            _nkWithDensePairMapped(a, b, map: { nk_f16_t($0.bitPattern) }) { ap, bp, n in
                var result: Float32 = 0
                nk_sqeuclidean_f16(ap, bp, UInt64(n), &result)
                return result
            }
        }
    }
#endif

extension Int8: NumKongDot {
    public typealias DotOutput = Int32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Int32?
    where A: Sequence, B: Sequence, A.Element == Int8, B.Element == Int8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Int32 = 0
            nk_dot_i8(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Int8: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == Int8, B.Element == Int8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float32 = 0
            nk_angular_i8(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Int8: NumKongEuclidean {
    public typealias L2Output = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == Int8, B.Element == Int8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float32 = 0
            nk_euclidean_i8(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Int8: NumKongSqeuclidean {
    public typealias L2sqOutput = UInt32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> UInt32?
    where A: Sequence, B: Sequence, A.Element == Int8, B.Element == Int8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: UInt32 = 0
            nk_sqeuclidean_i8(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension UInt8: NumKongDot {
    public typealias DotOutput = UInt32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> UInt32?
    where A: Sequence, B: Sequence, A.Element == UInt8, B.Element == UInt8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: UInt32 = 0
            nk_dot_u8(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension UInt8: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == UInt8, B.Element == UInt8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float32 = 0
            nk_angular_u8(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension UInt8: NumKongEuclidean {
    public typealias L2Output = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == UInt8, B.Element == UInt8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float32 = 0
            nk_euclidean_u8(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension UInt8: NumKongSqeuclidean {
    public typealias L2sqOutput = UInt32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> UInt32?
    where A: Sequence, B: Sequence, A.Element == UInt8, B.Element == UInt8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: UInt32 = 0
            nk_sqeuclidean_u8(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

// MARK: - Minifloat Scalars

extension BFloat16: NumKongDot {
    public typealias DotOutput = Float32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == BFloat16, B.Element == BFloat16 {
        _nkWithDensePairMapped(a, b, map: { nk_bf16_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_dot_bf16(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension BFloat16: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == BFloat16, B.Element == BFloat16 {
        _nkWithDensePairMapped(a, b, map: { nk_bf16_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_angular_bf16(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension BFloat16: NumKongEuclidean {
    public typealias L2Output = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == BFloat16, B.Element == BFloat16 {
        _nkWithDensePairMapped(a, b, map: { nk_bf16_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_euclidean_bf16(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension BFloat16: NumKongSqeuclidean {
    public typealias L2sqOutput = Float32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == BFloat16, B.Element == BFloat16 {
        _nkWithDensePairMapped(a, b, map: { nk_bf16_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_sqeuclidean_bf16(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E4M3: NumKongDot {
    public typealias DotOutput = Float32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E4M3, B.Element == E4M3 {
        _nkWithDensePairMapped(a, b, map: { nk_e4m3_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_dot_e4m3(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E4M3: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E4M3, B.Element == E4M3 {
        _nkWithDensePairMapped(a, b, map: { nk_e4m3_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_angular_e4m3(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E4M3: NumKongEuclidean {
    public typealias L2Output = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E4M3, B.Element == E4M3 {
        _nkWithDensePairMapped(a, b, map: { nk_e4m3_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_euclidean_e4m3(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E4M3: NumKongSqeuclidean {
    public typealias L2sqOutput = Float32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E4M3, B.Element == E4M3 {
        _nkWithDensePairMapped(a, b, map: { nk_e4m3_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_sqeuclidean_e4m3(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E5M2: NumKongDot {
    public typealias DotOutput = Float32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E5M2, B.Element == E5M2 {
        _nkWithDensePairMapped(a, b, map: { nk_e5m2_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_dot_e5m2(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E5M2: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E5M2, B.Element == E5M2 {
        _nkWithDensePairMapped(a, b, map: { nk_e5m2_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_angular_e5m2(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E5M2: NumKongEuclidean {
    public typealias L2Output = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E5M2, B.Element == E5M2 {
        _nkWithDensePairMapped(a, b, map: { nk_e5m2_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_euclidean_e5m2(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E5M2: NumKongSqeuclidean {
    public typealias L2sqOutput = Float32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E5M2, B.Element == E5M2 {
        _nkWithDensePairMapped(a, b, map: { nk_e5m2_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_sqeuclidean_e5m2(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E2M3: NumKongDot {
    public typealias DotOutput = Float32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E2M3, B.Element == E2M3 {
        _nkWithDensePairMapped(a, b, map: { nk_e2m3_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_dot_e2m3(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E2M3: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E2M3, B.Element == E2M3 {
        _nkWithDensePairMapped(a, b, map: { nk_e2m3_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_angular_e2m3(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E2M3: NumKongEuclidean {
    public typealias L2Output = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E2M3, B.Element == E2M3 {
        _nkWithDensePairMapped(a, b, map: { nk_e2m3_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_euclidean_e2m3(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E2M3: NumKongSqeuclidean {
    public typealias L2sqOutput = Float32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E2M3, B.Element == E2M3 {
        _nkWithDensePairMapped(a, b, map: { nk_e2m3_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_sqeuclidean_e2m3(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E3M2: NumKongDot {
    public typealias DotOutput = Float32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E3M2, B.Element == E3M2 {
        _nkWithDensePairMapped(a, b, map: { nk_e3m2_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_dot_e3m2(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E3M2: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E3M2, B.Element == E3M2 {
        _nkWithDensePairMapped(a, b, map: { nk_e3m2_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_angular_e3m2(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E3M2: NumKongEuclidean {
    public typealias L2Output = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E3M2, B.Element == E3M2 {
        _nkWithDensePairMapped(a, b, map: { nk_e3m2_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_euclidean_e3m2(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E3M2: NumKongSqeuclidean {
    public typealias L2sqOutput = Float32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E3M2, B.Element == E3M2 {
        _nkWithDensePairMapped(a, b, map: { nk_e3m2_t($0.bitPattern) }) { ap, bp, n in
            var result: Float32 = 0
            nk_sqeuclidean_e3m2(ap, bp, UInt64(n), &result)
            return result
        }
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

extension RandomAccessCollection where Element: NumKongEuclidean {
    @inlinable @inline(__always)
    public func euclidean<B>(_ b: B) -> Element.L2Output?
    where B: Sequence, B.Element == Element {
        Element.euclidean(self, b)
    }
}

extension RandomAccessCollection where Element: NumKongSqeuclidean {
    @inlinable @inline(__always)
    public func sqeuclidean<B>(_ b: B) -> Element.L2sqOutput?
    where B: Sequence, B.Element == Element {
        Element.sqeuclidean(self, b)
    }
}
