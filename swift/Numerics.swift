import CNumKong

// MARK: - Shared Helpers

@usableFromInline
func _nkWithDensePair<A: Sequence, B: Sequence, T, R>(
    _ a: A,
    _ b: B,
    _ body: (UnsafePointer<T>, UnsafePointer<T>, Int) -> R
) -> R?
where A.Element == T, B.Element == T {
    let lhs = Array(a)
    let rhs = Array(b)
    guard !lhs.isEmpty && lhs.count == rhs.count else { return nil }
    return lhs.withUnsafeBufferPointer { lhsPtr in
        rhs.withUnsafeBufferPointer { rhsPtr in
            body(lhsPtr.baseAddress!, rhsPtr.baseAddress!, lhs.count)
        }
    }
}

@usableFromInline
func _nkWithDensePairMapped<A: Sequence, B: Sequence, E, T, R>(
    _ a: A,
    _ b: B,
    map: (E) -> T,
    _ body: (UnsafePointer<T>, UnsafePointer<T>, Int) -> R
) -> R?
where A.Element == E, B.Element == E {
    let lhs = Array(a).map(map)
    let rhs = Array(b).map(map)
    guard !lhs.isEmpty && lhs.count == rhs.count else { return nil }
    return lhs.withUnsafeBufferPointer { lhsPtr in
        rhs.withUnsafeBufferPointer { rhsPtr in
            body(lhsPtr.baseAddress!, rhsPtr.baseAddress!, lhs.count)
        }
    }
}

@usableFromInline
func _nkF32ToBf16Bits(_ value: Float32) -> UInt16 {
    var src = value
    var dst: nk_bf16_t = 0
    nk_f32_to_bf16(&src, &dst)
    return UInt16(dst)
}

@usableFromInline
func _nkBf16BitsToF32(_ bits: UInt16) -> Float32 {
    var src = nk_bf16_t(bits)
    var dst: Float32 = 0
    nk_bf16_to_f32(&src, &dst)
    return dst
}

@usableFromInline
func _nkF32ToE4M3Bits(_ value: Float32) -> UInt8 {
    var src = value
    var dst: nk_e4m3_t = 0
    nk_f32_to_e4m3(&src, &dst)
    return UInt8(dst)
}

@usableFromInline
func _nkE4M3BitsToF32(_ bits: UInt8) -> Float32 {
    var src = nk_e4m3_t(bits)
    var dst: Float32 = 0
    nk_e4m3_to_f32(&src, &dst)
    return dst
}

@usableFromInline
func _nkF32ToE5M2Bits(_ value: Float32) -> UInt8 {
    var src = value
    var dst: nk_e5m2_t = 0
    nk_f32_to_e5m2(&src, &dst)
    return UInt8(dst)
}

@usableFromInline
func _nkE5M2BitsToF32(_ bits: UInt8) -> Float32 {
    var src = nk_e5m2_t(bits)
    var dst: Float32 = 0
    nk_e5m2_to_f32(&src, &dst)
    return dst
}

@usableFromInline
func _nkF32ToE2M3Bits(_ value: Float32) -> UInt8 {
    var src = value
    var dst: nk_e2m3_t = 0
    nk_f32_to_e2m3(&src, &dst)
    return UInt8(dst)
}

@usableFromInline
func _nkE2M3BitsToF32(_ bits: UInt8) -> Float32 {
    var src = nk_e2m3_t(bits)
    var dst: Float32 = 0
    nk_e2m3_to_f32(&src, &dst)
    return dst
}

@usableFromInline
func _nkF32ToE3M2Bits(_ value: Float32) -> UInt8 {
    var src = value
    var dst: nk_e3m2_t = 0
    nk_f32_to_e3m2(&src, &dst)
    return UInt8(dst)
}

@usableFromInline
func _nkE3M2BitsToF32(_ bits: UInt8) -> Float32 {
    var src = nk_e3m2_t(bits)
    var dst: Float32 = 0
    nk_e3m2_to_f32(&src, &dst)
    return dst
}

// MARK: - Low-Precision Storage Types

public struct BFloat16: Equatable, Hashable, Sendable {
    public var bitPattern: UInt16

    @inlinable
    public init(bitPattern: UInt16) {
        self.bitPattern = bitPattern
    }

    @inlinable
    public init(float: Float32) {
        self.bitPattern = _nkF32ToBf16Bits(float)
    }

    @inlinable
    public var float: Float32 { _nkBf16BitsToF32(bitPattern) }
}

public struct E4M3: Equatable, Hashable, Sendable {
    public var bitPattern: UInt8

    @inlinable
    public init(bitPattern: UInt8) {
        self.bitPattern = bitPattern
    }

    @inlinable
    public init(float: Float32) {
        self.bitPattern = _nkF32ToE4M3Bits(float)
    }

    @inlinable
    public var float: Float32 { _nkE4M3BitsToF32(bitPattern) }
}

public struct E5M2: Equatable, Hashable, Sendable {
    public var bitPattern: UInt8

    @inlinable
    public init(bitPattern: UInt8) {
        self.bitPattern = bitPattern
    }

    @inlinable
    public init(float: Float32) {
        self.bitPattern = _nkF32ToE5M2Bits(float)
    }

    @inlinable
    public var float: Float32 { _nkE5M2BitsToF32(bitPattern) }
}

public struct E2M3: Equatable, Hashable, Sendable {
    public var bitPattern: UInt8

    @inlinable
    public init(bitPattern: UInt8) {
        self.bitPattern = bitPattern
    }

    @inlinable
    public init(float: Float32) {
        self.bitPattern = _nkF32ToE2M3Bits(float)
    }

    @inlinable
    public var float: Float32 { _nkE2M3BitsToF32(bitPattern) }
}

public struct E3M2: Equatable, Hashable, Sendable {
    public var bitPattern: UInt8

    @inlinable
    public init(bitPattern: UInt8) {
        self.bitPattern = bitPattern
    }

    @inlinable
    public init(float: Float32) {
        self.bitPattern = _nkF32ToE3M2Bits(float)
    }

    @inlinable
    public var float: Float32 { _nkE3M2BitsToF32(bitPattern) }
}
