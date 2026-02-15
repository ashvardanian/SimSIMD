# NumKong for Swift

To install, simply add the following dependency to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/ashvardanian/numkong")
]
```

The package provides the most common spatial metrics for `Int8`, `Float16`, `Float32`, and `Float64` vectors.

```swift
import NumKong

let vectorA: [Int8] = [1, 2, 3]
let vectorB: [Int8] = [4, 5, 6]

let dotProduct = vectorA.dot(vectorB)           // Computes the dot product
let angularDistance = vectorA.angular(vectorB)  // Computes the angular distance
let sqEuclidean = vectorA.sqeuclidean(vectorB)  // Computes the squared Euclidean distance
```
