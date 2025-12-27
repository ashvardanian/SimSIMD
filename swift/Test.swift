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
        XCTAssertEqual(result, 32.0, accuracy: 0.01)
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
        XCTAssertEqual(result, 27.0, accuracy: 0.01)
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
}
