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
    // ARM NEON
    if contains(.neon) { components.append(".neon") }
    if contains(.neonhalf) { components.append(".neonhalf") }
    if contains(.neonfhm) { components.append(".neonfhm") }
    if contains(.neonbfdot) { components.append(".neonbfdot") }
    if contains(.neonsdot) { components.append(".neonsdot") }
    // ARM SVE
    if contains(.sve) { components.append(".sve") }
    if contains(.svehalf) { components.append(".svehalf") }
    if contains(.svebfdot) { components.append(".svebfdot") }
    if contains(.svesdot) { components.append(".svesdot") }
    if contains(.sve2) { components.append(".sve2") }
    if contains(.sve2p1) { components.append(".sve2p1") }
    // ARM SME
    if contains(.sme) { components.append(".sme") }
    if contains(.sme2) { components.append(".sme2") }
    if contains(.sme2p1) { components.append(".sme2p1") }
    if contains(.smef64) { components.append(".smef64") }
    if contains(.smehalf) { components.append(".smehalf") }
    if contains(.smebf16) { components.append(".smebf16") }
    if contains(.smelut2) { components.append(".smelut2") }
    if contains(.smefa64) { components.append(".smefa64") }
    // x86
    if contains(.haswell) { components.append(".haswell") }
    if contains(.skylake) { components.append(".skylake") }
    if contains(.ice) { components.append(".ice") }
    if contains(.genoa) { components.append(".genoa") }
    if contains(.sapphire) { components.append(".sapphire") }
    if contains(.sapphireAmx) { components.append(".sapphireAmx") }
    if contains(.graniteAmx) { components.append(".graniteAmx") }
    if contains(.turin) { components.append(".turin") }
    if contains(.sierra) { components.append(".sierra") }
    return "[\(components.joined(separator: ", "))]"
  }

  public static let available = nk_capabilities()

  public static let any = nk_cap_any_k
  // ARM NEON
  public static let neon = nk_cap_neon_k
  public static let neonhalf = nk_cap_neonhalf_k
  public static let neonfhm = nk_cap_neonfhm_k
  public static let neonbfdot = nk_cap_neonbfdot_k
  public static let neonsdot = nk_cap_neonsdot_k
  // ARM SVE
  public static let sve = nk_cap_sve_k
  public static let svehalf = nk_cap_svehalf_k
  public static let svebfdot = nk_cap_svebfdot_k
  public static let svesdot = nk_cap_svesdot_k
  public static let sve2 = nk_cap_sve2_k
  public static let sve2p1 = nk_cap_sve2p1_k
  // ARM SME
  public static let sme = nk_cap_sme_k
  public static let sme2 = nk_cap_sme2_k
  public static let sme2p1 = nk_cap_sme2p1_k
  public static let smef64 = nk_cap_smef64_k
  public static let smehalf = nk_cap_smehalf_k
  public static let smebf16 = nk_cap_smebf16_k
  public static let smelut2 = nk_cap_smelut2_k
  public static let smefa64 = nk_cap_smefa64_k
  // x86
  public static let haswell = nk_cap_haswell_k
  public static let skylake = nk_cap_skylake_k
  public static let ice = nk_cap_ice_k
  public static let genoa = nk_cap_genoa_k
  public static let sapphire = nk_cap_sapphire_k
  public static let sapphireAmx = nk_cap_sapphire_amx_k
  public static let graniteAmx = nk_cap_granite_amx_k
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
