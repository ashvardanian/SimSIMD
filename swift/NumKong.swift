import CNumKong

public protocol NumKong {
  static var dataType: nk_datatype_t { get }
  static var angular: nk_dense_metric_t { get }
  static var dotProduct: nk_dense_metric_t { get }
  static var euclidean: nk_dense_metric_t { get }
  static var squaredEuclidean: nk_dense_metric_t { get }
}

extension Int8: NumKong {
  public static let dataType = nk_i8_k
  public static let angular = find(kind: nk_angular_k, dataType: dataType)
  public static let dotProduct = find(kind: nk_dot_k, dataType: dataType)
  public static let euclidean = find(kind: nk_euclidean_k, dataType: dataType)
  public static let squaredEuclidean = find(kind: nk_sqeuclidean_k, dataType: dataType)
}

#if !arch(x86_64)
  @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
  extension Float16: NumKong {
    public static let dataType = nk_f16_k
    public static let angular = find(kind: nk_angular_k, dataType: dataType)
    public static let dotProduct = find(kind: nk_dot_k, dataType: dataType)
    public static let euclidean = find(kind: nk_euclidean_k, dataType: dataType)
    public static let squaredEuclidean = find(kind: nk_sqeuclidean_k, dataType: dataType)
  }
#endif

extension Float32: NumKong {
  public static let dataType = nk_f32_k
  public static let angular = find(kind: nk_angular_k, dataType: dataType)
  public static let dotProduct = find(kind: nk_inner_k, dataType: dataType)
  public static let euclidean = find(kind: nk_euclidean_k, dataType: dataType)
  public static let squaredEuclidean = find(kind: nk_sqeuclidean_k, dataType: dataType)
}

extension Float64: NumKong {
  public static let dataType = nk_f64_k
  public static let angular = find(kind: nk_angular_k, dataType: dataType)
  public static let dotProduct = find(kind: nk_dot_k, dataType: dataType)
  public static let euclidean = find(kind: nk_euclidean_k, dataType: dataType)
  public static let squaredEuclidean = find(kind: nk_sqeuclidean_k, dataType: dataType)
}

extension NumKong {
  @inlinable @inline(__always)
  public static func angular<A, B>(_ a: A, _ b: B) -> Double?
  where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self {
    perform(angular, a: a, b: b)
  }

  @inlinable @inline(__always)
  public static func dot<A, B>(_ a: A, _ b: B) -> Double?
  where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self {
    perform(dotProduct, a: a, b: b)
  }

  @inlinable @inline(__always)
  public static func euclidean<A, B>(_ a: A, _ b: B) -> Double?
  where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self {
    perform(euclidean, a: a, b: b)
  }

  @inlinable @inline(__always)
  public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Double?
  where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self {
    perform(squaredEuclidean, a: a, b: b)
  }
}

extension RandomAccessCollection where Element: NumKong {
  @inlinable @inline(__always)
  public func angular<B>(_ b: B) -> Double? where B: Sequence, B.Element == Element {
    Element.angular(self, b)
  }

  @inlinable @inline(__always)
  public func dot<B>(_ b: B) -> Double? where B: Sequence, B.Element == Element {
    Element.dot(self, b)
  }

  @inlinable @inline(__always)
  public func euclidean<B>(_ b: B) -> Double? where B: Sequence, B.Element == Element {
    Element.euclidean(self, b)
  }

  @inlinable @inline(__always)
  public func sqeuclidean<B>(_ b: B) -> Double? where B: Sequence, B.Element == Element {
    Element.sqeuclidean(self, b)
  }
}

@inlinable @inline(__always)
func perform<A, B>(_ metric: nk_dense_metric_t, a: A, b: B) -> Double?
where A: Sequence, B: Sequence, A.Element == B.Element {
  var distance: nk_fmax_t = 0
  let result = a.withContiguousStorageIfAvailable { a in
    b.withContiguousStorageIfAvailable { b in
      guard a.count > 0 && a.count == b.count else { return false }
      metric(a.baseAddress, b.baseAddress, .init(a.count), &distance)
      return true
    }
  }
  guard result == true else { return nil }
  return distance
}

public typealias Capabilities = nk_capability_t

extension nk_capability_t: OptionSet, CustomStringConvertible {
  public var description: String {
    var components: [String] = []
    if contains(.neon) { components.append(".neon") }
    if contains(.sve) { components.append(".sve") }
    if contains(.sve2) { components.append(".sve2") }
    if contains(.haswell) { components.append(".haswell") }
    if contains(.skylake) { components.append(".skylake") }
    if contains(.ice) { components.append(".ice") }
    if contains(.genoa) { components.append(".genoa") }
    if contains(.sapphire) { components.append(".sapphire") }
    if contains(.turin) { components.append(".turin") }
    if contains(.sierra) { components.append(".sierra") }
    return "[\(components.joined(separator: ", "))]"
  }

  public static let available = nk_capabilities()

  public static let any = nk_cap_any_k
  public static let neon = nk_cap_neon_k
  public static let sve = nk_cap_sve_k
  public static let sve2 = nk_cap_sve2_k
  public static let haswell = nk_cap_haswell_k
  public static let skylake = nk_cap_skylake_k
  public static let ice = nk_cap_ice_k
  public static let genoa = nk_cap_genoa_k
  public static let sapphire = nk_cap_sapphire_k
  public static let turin = nk_cap_turin_k
  public static let sierra = nk_cap_sierra_k
}

@inline(__always)
private func find(kind: nk_kernel_kind_t, dataType: nk_datatype_t)
  -> nk_dense_metric_t
{
  var output: nk_dense_metric_t?
  var used = nk_capability_t.any
  // Use `withUnsafeMutablePointer` to safely cast `output` to the required pointer type.
  withUnsafeMutablePointer(to: &output) { outputPtr in
    // Cast the pointer to `UnsafeMutablePointer<nk_kernel_punned_t?>`
    let castedPtr = outputPtr.withMemoryRebound(
      to: Optional<nk_kernel_punned_t>.self, capacity: 1
    ) { $0 }
    nk_find_kernel(kind, dataType, .available, .any, castedPtr, &used)
  }
  guard let output else { fatalError("Could not find function \(kind) for \(dataType)") }
  return output
}
