# NumKong for Rust


## Installation

To install, add the following to your `Cargo.toml`:

```toml
[dependencies]
numkong = "..."
```

Before using the NumKong library, ensure you have imported the necessary traits and types into your Rust source file.
The library provides several traits for different distance/similarity kinds - `SpatialSimilarity`, `BinarySimilarity`, and `ProbabilitySimilarity`.

## Spatial Similarity: Angular and Euclidean Distances

```rust
use numkong::SpatialSimilarity;

fn main() {
    let vector_a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let vector_b: Vec<f32> = vec![4.0, 5.0, 6.0];

    // Compute the angular distance between vectors
    let angular_distance = f32::angular(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Angular Distance: {}", angular_distance);

    // Compute the squared Euclidean distance between vectors
    let sq_euclidean_distance = f32::sqeuclidean(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Squared Euclidean Distance: {}", sq_euclidean_distance);
}
```

Spatial similarity functions are available for `f64`, `f32`, `f16`, and `i8` types.

## Dot-Products: Inner and Complex Inner Products

```rust
use numkong::SpatialSimilarity;
use numkong::ComplexProducts;

fn main() {
    // Complex vectors have interleaved real & imaginary components
    let vector_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let vector_b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

    // Compute the inner product between vectors
    let inner_product = SpatialSimilarity::dot(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Inner Product: {}", inner_product);

    // Compute the complex inner product between vectors
    let complex_inner_product = ComplexProducts::dot(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    let complex_conjugate_inner_product = ComplexProducts::vdot(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Complex Inner Product: {:?}", complex_inner_product); // -18, 69
    println!("Complex C. Inner Product: {:?}", complex_conjugate_inner_product); // 70, -8
}
```

Complex inner products are available for `f64`, `f32`, and `f16` types.

## Probability Distributions: Jensen-Shannon and Kullback-Leibler Divergences

```rust
use numkong::ProbabilitySimilarity;

fn main() {
    let vector_a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let vector_b: Vec<f32> = vec![4.0, 5.0, 6.0];

    let jensen_shannon = f32::jensenshannon(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Jensen-Shannon Divergence: {}", jensen_shannon);

    let kullback_leibler = f32::kullbackleibler(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Kullback-Leibler Divergence: {}", kullback_leibler);
}
```

Probability similarity functions are available for `f64`, `f32`, and `f16` types.

## Binary Similarity: Hamming and Jaccard Distances

Similar to spatial distances, one can compute bit-level distance functions between slices of unsigned integers:

```rust
use numkong::BinarySimilarity;

fn main() {
    let vector_a = &[0b11110000, 0b00001111, 0b10101010];
    let vector_b = &[0b11110000, 0b00001111, 0b01010101];

    // Compute the Hamming distance between vectors
    let hamming_distance = u8::hamming(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Hamming Distance: {}", hamming_distance);

    // Compute the Jaccard distance between vectors
    let jaccard_distance = u8::jaccard(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Jaccard Distance: {}", jaccard_distance);
}
```

Binary similarity functions are available only for `u8` types.

## Half-Precision Floating-Point Numbers

Rust has no native support for half-precision floating-point numbers, but NumKong provides a `f16` type with built-in conversion methods.
The underlying `u16` representation is publicly accessible for direct bit manipulation.

```rust
use numkong::{SpatialSimilarity, f16};

fn main() {
    // Create f16 vectors using built-in conversion methods
    let vector_a: Vec<f16> = vec![1.0, 2.0, 3.0].iter().map(|&x| f16::from_f32(x)).collect();
    let vector_b: Vec<f16> = vec![4.0, 5.0, 6.0].iter().map(|&x| f16::from_f32(x)).collect();

    // Compute the angular distance
    let angular_distance = f16::angular(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Angular Distance: {}", angular_distance);

    // Direct bit manipulation
    let half = f16::from_f32(3.14159);
    let bits = half.0; // Access raw u16 representation
    let reconstructed = f16(bits);
    
    // Convert back to f32
    let float_value = half.to_f32();
}
```

For interoperability with the `half` crate:

```rust
use numkong::{SpatialSimilarity, f16 as SimF16};
use half::f16 as HalfF16;

fn main() {
    let vector_a: Vec<HalfF16> = vec![1.0, 2.0, 3.0].iter().map(|&x| HalfF16::from_f32(x)).collect();
    let vector_b: Vec<HalfF16> = vec![4.0, 5.0, 6.0].iter().map(|&x| HalfF16::from_f32(x)).collect();

    // Safe reinterpret cast due to identical memory layout
    let buffer_a: &[SimF16] = unsafe { std::slice::from_raw_parts(vector_a.as_ptr() as *const SimF16, vector_a.len()) };
    let buffer_b: &[SimF16] = unsafe { std::slice::from_raw_parts(vector_b.as_ptr() as *const SimF16, vector_b.len()) };

    let angular_distance = SimF16::angular(buffer_a, buffer_b)
        .expect("Vectors must be of the same length");

    println!("Angular Distance: {}", angular_distance);
}
```

## Half-Precision Brain-Float Numbers

The "brain-float-16" is a popular machine learning format.
It's broadly supported in hardware and is very machine-friendly, but software support is still lagging behind.
[Unlike NumPy](https://github.com/numpy/numpy/issues/19808), you can already use `bf16` datatype in NumKong.
NumKong provides a `bf16` type with built-in conversion methods and direct bit access.

```rust
use numkong::{SpatialSimilarity, bf16};

fn main() {
    // Create bf16 vectors using built-in conversion methods
    let vector_a: Vec<bf16> = vec![1.0, 2.0, 3.0].iter().map(|&x| bf16::from_f32(x)).collect();
    let vector_b: Vec<bf16> = vec![4.0, 5.0, 6.0].iter().map(|&x| bf16::from_f32(x)).collect();

    // Compute the angular distance
    let angular_distance = bf16::angular(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Angular Distance: {}", angular_distance);

    // Direct bit manipulation
    let brain_half = bf16::from_f32(3.14159);
    let bits = brain_half.0; // Access raw u16 representation
    let reconstructed = bf16(bits);
    
    // Convert back to f32
    let float_value = brain_half.to_f32();

    // Compare precision differences
    let original = 3.14159_f32;
    let f16_roundtrip = f16::from_f32(original).to_f32();
    let bf16_roundtrip = bf16::from_f32(original).to_f32();
    
    println!("Original: {}", original);
    println!("f16 roundtrip: {}", f16_roundtrip);
    println!("bf16 roundtrip: {}", bf16_roundtrip);
}
```

## Dynamic Dispatch in Rust

NumKong provides a [dynamic dispatch](#dynamic-dispatch) mechanism to select the most advanced micro-kernel for the current CPU.
You can query supported backends and use the `NumKong::capabilities` function to select the best one.

```rust
println!("uses neon: {}", capabilities::uses_neon());
println!("uses sve: {}", capabilities::uses_sve());
println!("uses haswell: {}", capabilities::uses_haswell());
println!("uses skylake: {}", capabilities::uses_skylake());
println!("uses ice: {}", capabilities::uses_ice());
println!("uses genoa: {}", capabilities::uses_genoa());
println!("uses sapphire: {}", capabilities::uses_sapphire());
println!("uses turin: {}", capabilities::uses_turin());
println!("uses sierra: {}", capabilities::uses_sierra());
```
