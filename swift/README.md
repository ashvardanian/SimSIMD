# NumKong for Swift

Add NumKong to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/ashvardanian/numkong")
]
```

The Swift binding exposes:

- Vector APIs (`dot`, `angular`, `euclidean`, `sqeuclidean`)
- Packed matrix APIs (`dots_packed`, `angulars_packed`, `euclideans_packed`)
- Symmetric matrix APIs (`dots_symmetric`, `angulars_symmetric`, `euclideans_symmetric`)
- Low-precision numeric wrappers (`BFloat16`, `E4M3`, `E5M2`, `E2M3`, `E3M2`)

## Vector Example

```swift
import NumKong

let a: [Float32] = [1, 2, 3]
let b: [Float32] = [4, 5, 6]
let value = a.dot(b)
```

## Packed Matrix Example

```swift
import NumKong

let a: [Float32] = [1, 2, 3, 4, 5, 6] // 2x3
let b: [Float32] = [7, 8, 9, 1, 0, 1] // 2x3
var out = Array(repeating: Float32(0), count: 4) // 2x2

try a.withUnsafeBufferPointer { aPtr in
    try b.withUnsafeBufferPointer { bPtr in
        try out.withUnsafeMutableBufferPointer { outPtr in
            let aMatrix = Matrix(baseAddress: aPtr.baseAddress!, rows: 2, cols: 3)
            let bMatrix = Matrix(baseAddress: bPtr.baseAddress!, rows: 2, cols: 3)
            var outMatrix = MutableMatrix(baseAddress: outPtr.baseAddress!, rows: 2, cols: 2)
            let packed = try PackedMatrix<Float32>(packing: bMatrix)
            try dots_packed(aMatrix, packed, &outMatrix)
        }
    }
}
```
