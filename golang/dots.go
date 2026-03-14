package numkong

/*
#cgo CFLAGS: -O3 -I../include
#cgo LDFLAGS: -O3 -L. -lm
#define NK_NATIVE_F16 (0)
#define NK_NATIVE_BF16 (0)
#include "numkong/numkong.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

// region Packed Batch Operations (Dots)

// DotsPackedF64 computes A × Bᵀ where B is pre-packed. Output type: f64.
// a must have capacity >= height × b.Depth().
// c must have capacity >= height × b.Width().
func DotsPackedF64(a []float64, b PackedMatrix, c []float64, height int) {
	if b.Dtype() != "f64" {
		panic("PackedMatrix dtype must be f64")
	}
	if len(a) < height*b.depth {
		panic("input slice too short for the given height and depth")
	}
	if len(c) < height*b.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_dots_packed_f64(
		(*C.nk_f64_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_f64_t)(&c[0]),
		C.nk_size_t(height), C.nk_size_t(b.width), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth*8),
		C.nk_size_t(b.width*8))
}

// DotsPackedF32 computes A × Bᵀ where B is pre-packed. Output type: f64 (widened).
// a must have capacity >= height × b.Depth().
// c must have capacity >= height × b.Width().
func DotsPackedF32(a []float32, b PackedMatrix, c []float64, height int) {
	if b.Dtype() != "f32" {
		panic("PackedMatrix dtype must be f32")
	}
	if len(a) < height*b.depth {
		panic("input slice too short for the given height and depth")
	}
	if len(c) < height*b.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_dots_packed_f32(
		(*C.nk_f32_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_f64_t)(&c[0]),
		C.nk_size_t(height), C.nk_size_t(b.width), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth*4),
		C.nk_size_t(b.width*8))
}

// DotsPackedI8 computes A × Bᵀ where B is pre-packed. Output type: i32 (widened).
// a must have capacity >= height × b.Depth().
// c must have capacity >= height × b.Width().
func DotsPackedI8(a []int8, b PackedMatrix, c []int32, height int) {
	if b.Dtype() != "i8" {
		panic("PackedMatrix dtype must be i8")
	}
	if len(a) < height*b.depth {
		panic("input slice too short for the given height and depth")
	}
	if len(c) < height*b.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_dots_packed_i8(
		(*C.nk_i8_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_i32_t)(&c[0]),
		C.nk_size_t(height), C.nk_size_t(b.width), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth),
		C.nk_size_t(b.width*4))
}

// DotsPackedU8 computes A × Bᵀ where B is pre-packed. Output type: u32 (widened).
// a must have capacity >= height × b.Depth().
// c must have capacity >= height × b.Width().
func DotsPackedU8(a []uint8, b PackedMatrix, c []uint32, height int) {
	if b.Dtype() != "u8" {
		panic("PackedMatrix dtype must be u8")
	}
	if len(a) < height*b.depth {
		panic("input slice too short for the given height and depth")
	}
	if len(c) < height*b.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_dots_packed_u8(
		(*C.nk_u8_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_u32_t)(&c[0]),
		C.nk_size_t(height), C.nk_size_t(b.width), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth),
		C.nk_size_t(b.width*4))
}

// endregion

// region Symmetric Operations (Dots)

// DotsSymmetricF64 computes the Gram matrix (all-pairs dot products) for a set of f64 vectors.
// vectors: nVectors × depth row-major matrix.
// result: nVectors × nVectors output matrix (upper triangle filled, lower mirrored).
func DotsSymmetricF64(vectors []float64, nVectors, depth int, result []float64) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	dotsSymmetricF64(vectors, nVectors, depth, result, 0, nVectors)
}

func dotsSymmetricF64(vectors []float64, nVectors, depth int, result []float64, rowStart, rowCount int) {
	C.nk_dots_symmetric_f64(
		(*C.nk_f64_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth*8),
		(*C.nk_f64_t)(&result[0]),
		C.nk_size_t(nVectors*8),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}

// DotsSymmetricF32 computes the Gram matrix (all-pairs dot products) for a set of f32 vectors.
// vectors: nVectors × depth row-major matrix.
// result: nVectors × nVectors output matrix. Output: f64.
func DotsSymmetricF32(vectors []float32, nVectors, depth int, result []float64) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	dotsSymmetricF32(vectors, nVectors, depth, result, 0, nVectors)
}

func dotsSymmetricF32(vectors []float32, nVectors, depth int, result []float64, rowStart, rowCount int) {
	C.nk_dots_symmetric_f32(
		(*C.nk_f32_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth*4),
		(*C.nk_f64_t)(&result[0]),
		C.nk_size_t(nVectors*8),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}

// DotsSymmetricI8 computes the Gram matrix (all-pairs dot products) for a set of i8 vectors.
// vectors: nVectors × depth row-major matrix.
// result: nVectors × nVectors output matrix. Output: i32.
func DotsSymmetricI8(vectors []int8, nVectors, depth int, result []int32) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	dotsSymmetricI8(vectors, nVectors, depth, result, 0, nVectors)
}

func dotsSymmetricI8(vectors []int8, nVectors, depth int, result []int32, rowStart, rowCount int) {
	C.nk_dots_symmetric_i8(
		(*C.nk_i8_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth),
		(*C.nk_i32_t)(&result[0]),
		C.nk_size_t(nVectors*4),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}

// DotsSymmetricU8 computes the Gram matrix (all-pairs dot products) for a set of u8 vectors.
// vectors: nVectors × depth row-major matrix.
// result: nVectors × nVectors output matrix. Output: u32.
func DotsSymmetricU8(vectors []uint8, nVectors, depth int, result []uint32) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	dotsSymmetricU8(vectors, nVectors, depth, result, 0, nVectors)
}

func dotsSymmetricU8(vectors []uint8, nVectors, depth int, result []uint32, rowStart, rowCount int) {
	C.nk_dots_symmetric_u8(
		(*C.nk_u8_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth),
		(*C.nk_u32_t)(&result[0]),
		C.nk_size_t(nVectors*4),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}

// endregion
