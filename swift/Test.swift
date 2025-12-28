import NumKong
import XCTest

class NumKongTests: XCTestCase {
    override class func setUp() {
        print("Capabilities: \(Capabilities.available)")
    }

    func testAngularInt8() throws {
        let a: [Int8] = [3, 97, 127]
        let b: [Int8] = [3, 97, 127]
        let result = try XCTUnwrap(a.angular(b))
        XCTAssertEqual(result, 0.00012027938, accuracy: 0.01)
    }

    #if !arch(x86_64)
        func testAngularFloat16() throws {
            let a: [Float16] = [1.0, 2.0, 3.0]
            let b: [Float16] = [1.0, 2.0, 3.0]
            let result = try XCTUnwrap(a.angular(b))
            XCTAssertEqual(result, 0.004930496, accuracy: 0.01)
        }
    #endif

    func testAngularFloat32() throws {
        let a: [Float32] = [1.0, 2.0, 3.0]
        let b: [Float32] = [1.0, 2.0, 3.0]
        let result = try XCTUnwrap(a.angular(b))
        XCTAssertEqual(result, 0.004930496, accuracy: 0.01)
    }

    func testAngularFloat64() throws {
        let a: [Float64] = [1.0, 2.0, 3.0]
        let b: [Float64] = [1.0, 2.0, 3.0]
        let result = try XCTUnwrap(a.angular(b))
        XCTAssertEqual(result, 0.004930496, accuracy: 0.01)
    }

    func testInnerInt8() throws {
        let a: [Int8] = [1, 2, 3]
        let b: [Int8] = [4, 5, 6]
        let result = try XCTUnwrap(a.dot(b))
        XCTAssertEqual(result, 32)
    }

    #if !arch(x86_64)
        func testDotFloat16() throws {
            let a: [Float16] = [1.0, 2.0, 3.0]
            let b: [Float16] = [4.0, 5.0, 6.0]
            let result = try XCTUnwrap(a.dot(b))
            XCTAssertEqual(result, 32.0, accuracy: 0.01)
        }
    #endif

    func testDotFloat32() throws {
        let a: [Float32] = [1.0, 2.0, 3.0]
        let b: [Float32] = [4.0, 5.0, 6.0]
        let result = try XCTUnwrap(a.dot(b))
        XCTAssertEqual(result, 32.0, accuracy: 0.01)
    }

    func testDotFloat64() throws {
        let a: [Float64] = [1.0, 2.0, 3.0]
        let b: [Float64] = [4.0, 5.0, 6.0]
        let result = try XCTUnwrap(a.dot(b))
        XCTAssertEqual(result, 32.0, accuracy: 0.01)
    }

    func testEuclideanInt8() throws {
        let a: [Int8] = [1, 2, 3]
        let b: [Int8] = [4, 5, 6]
        let result = try XCTUnwrap(a.euclidean(b))
        XCTAssertEqual(result, 5.196152422706632, accuracy: 0.01)
    }

    #if !arch(x86_64)
        func testEuclideanFloat16() throws {
            let a: [Float16] = [1.0, 2.0, 3.0]
            let b: [Float16] = [4.0, 5.0, 6.0]
            let result = try XCTUnwrap(a.euclidean(b))
            XCTAssertEqual(result, 5.196152422706632, accuracy: 0.01)
        }
    #endif

    func testEuclideanFloat32() throws {
        let a: [Float32] = [1.0, 2.0, 3.0]
        let b: [Float32] = [4.0, 5.0, 6.0]
        let result = try XCTUnwrap(a.euclidean(b))
        XCTAssertEqual(result, 5.196152422706632, accuracy: 0.01)
    }

    func testEuclideanFloat64() throws {
        let a: [Float64] = [1.0, 2.0, 3.0]
        let b: [Float64] = [4.0, 5.0, 6.0]
        let result = try XCTUnwrap(a.euclidean(b))
        XCTAssertEqual(result, 5.196152422706632, accuracy: 0.01)
    }

    func testSqeuclideanInt8() throws {
        let a: [Int8] = [1, 2, 3]
        let b: [Int8] = [4, 5, 6]
        let result = try XCTUnwrap(a.sqeuclidean(b))
        XCTAssertEqual(result, 27)
    }

    #if !arch(x86_64)
        func testSqeuclideanFloat16() throws {
            let a: [Float16] = [1.0, 2.0, 3.0]
            let b: [Float16] = [4.0, 5.0, 6.0]
            let result = try XCTUnwrap(a.sqeuclidean(b))
            XCTAssertEqual(result, 27.0, accuracy: 0.01)
        }
    #endif

    func testSqeuclideanFloat32() throws {
        let a: [Float32] = [1.0, 2.0, 3.0]
        let b: [Float32] = [4.0, 5.0, 6.0]
        let result = try XCTUnwrap(a.sqeuclidean(b))
        XCTAssertEqual(result, 27.0, accuracy: 0.01)
    }

    func testSqeuclideanFloat64() throws {
        let a: [Float64] = [1.0, 2.0, 3.0]
        let b: [Float64] = [4.0, 5.0, 6.0]
        let result = try XCTUnwrap(a.sqeuclidean(b))
        XCTAssertEqual(result, 27.0, accuracy: 0.01)
    }

    // MARK: - Geospatial Tests

    func testHaversineFloat64() throws {
        // New York City to London (coordinates in radians)
        // NYC: 40.7128° N, 74.0060° W -> radians
        // London: 51.5074° N, 0.1278° W -> radians
        let aLat: [Float64] = [40.7128 * .pi / 180.0]
        let aLon: [Float64] = [-74.0060 * .pi / 180.0]
        let bLat: [Float64] = [51.5074 * .pi / 180.0]
        let bLon: [Float64] = [-0.1278 * .pi / 180.0]
        var result: [Float64] = [0]

        let success = aLat.withUnsafeBufferPointer { aLatPtr in
            aLon.withUnsafeBufferPointer { aLonPtr in
                bLat.withUnsafeBufferPointer { bLatPtr in
                    bLon.withUnsafeBufferPointer { bLonPtr in
                        result.withUnsafeMutableBufferPointer { resultPtr in
                            Float64.haversine(
                                aLat: aLatPtr,
                                aLon: aLonPtr,
                                bLat: bLatPtr,
                                bLon: bLonPtr,
                                result: resultPtr
                            )
                        }
                    }
                }
            }
        }
        XCTAssertTrue(success)
        // Expected distance: ~5539 km (Haversine formula)
        XCTAssertEqual(result[0], 5_539_000, accuracy: 5000)
    }

    func testHaversineFloat32() throws {
        let aLat: [Float32] = [40.7128 * .pi / 180.0]
        let aLon: [Float32] = [-74.0060 * .pi / 180.0]
        let bLat: [Float32] = [51.5074 * .pi / 180.0]
        let bLon: [Float32] = [-0.1278 * .pi / 180.0]
        var result: [Float32] = [0]

        let success = aLat.withUnsafeBufferPointer { aLatPtr in
            aLon.withUnsafeBufferPointer { aLonPtr in
                bLat.withUnsafeBufferPointer { bLatPtr in
                    bLon.withUnsafeBufferPointer { bLonPtr in
                        result.withUnsafeMutableBufferPointer { resultPtr in
                            Float32.haversine(
                                aLat: aLatPtr,
                                aLon: aLonPtr,
                                bLat: bLatPtr,
                                bLon: bLonPtr,
                                result: resultPtr
                            )
                        }
                    }
                }
            }
        }
        XCTAssertTrue(success)
        XCTAssertEqual(result[0], 5_539_000, accuracy: 5000)
    }

    func testVincentyFloat64() throws {
        let aLat: [Float64] = [40.7128 * .pi / 180.0]
        let aLon: [Float64] = [-74.0060 * .pi / 180.0]
        let bLat: [Float64] = [51.5074 * .pi / 180.0]
        let bLon: [Float64] = [-0.1278 * .pi / 180.0]
        var result: [Float64] = [0]

        let success = aLat.withUnsafeBufferPointer { aLatPtr in
            aLon.withUnsafeBufferPointer { aLonPtr in
                bLat.withUnsafeBufferPointer { bLatPtr in
                    bLon.withUnsafeBufferPointer { bLonPtr in
                        result.withUnsafeMutableBufferPointer { resultPtr in
                            Float64.vincenty(
                                aLat: aLatPtr,
                                aLon: aLonPtr,
                                bLat: bLatPtr,
                                bLon: bLonPtr,
                                result: resultPtr
                            )
                        }
                    }
                }
            }
        }
        XCTAssertTrue(success)
        // Expected distance: ~5570 km (Vincenty is more accurate)
        XCTAssertEqual(result[0], 5_570_000, accuracy: 20000)
    }

    func testVincentyFloat32() throws {
        let aLat: [Float32] = [40.7128 * .pi / 180.0]
        let aLon: [Float32] = [-74.0060 * .pi / 180.0]
        let bLat: [Float32] = [51.5074 * .pi / 180.0]
        let bLon: [Float32] = [-0.1278 * .pi / 180.0]
        var result: [Float32] = [0]

        let success = aLat.withUnsafeBufferPointer { aLatPtr in
            aLon.withUnsafeBufferPointer { aLonPtr in
                bLat.withUnsafeBufferPointer { bLatPtr in
                    bLon.withUnsafeBufferPointer { bLonPtr in
                        result.withUnsafeMutableBufferPointer { resultPtr in
                            Float32.vincenty(
                                aLat: aLatPtr,
                                aLon: aLonPtr,
                                bLat: bLatPtr,
                                bLon: bLonPtr,
                                result: resultPtr
                            )
                        }
                    }
                }
            }
        }
        XCTAssertTrue(success)
        XCTAssertEqual(result[0], 5_570_000, accuracy: 20000)
    }
}
