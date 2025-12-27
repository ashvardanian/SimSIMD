package numkong

/*
#cgo CFLAGS: -O3
#cgo LDFLAGS: -O3 -L. -lm
#define NK_NATIVE_F16 (0)
#define NK_NATIVE_BF16 (0)
#include "../include/numkong/numkong.h"
#include <stdlib.h>

inline static nk_f32_t cosine_i8(nk_i8_t const* a, nk_i8_t const* b, nk_size_t d) { return nk_metric_punned(nk_angular_k, nk_i8_k, nk_cap_any_k)(a, b, d, d); }
inline static nk_f32_t cosine_f32(nk_f32_t const* a, nk_f32_t const* b, nk_size_t d) { return nk_metric_punned(nk_angular_k, nk_f32_k, nk_cap_any_k)(a, b, d, d); }
inline static nk_f32_t inner_i8(nk_i8_t const* a, nk_i8_t const* b, nk_size_t d) { return nk_metric_punned(nk_inner_k, nk_i8_k, nk_cap_any_k)(a, b, d, d); }
inline static nk_f32_t inner_f32(nk_f32_t const* a, nk_f32_t const* b, nk_size_t d) { return nk_metric_punned(nk_inner_k, nk_f32_k, nk_cap_any_k)(a, b, d, d); }
inline static nk_f32_t sqeuclidean_i8(nk_i8_t const* a, nk_i8_t const* b, nk_size_t d) { return nk_metric_punned(nk_sqeuclidean_k, nk_i8_k, nk_cap_any_k)(a, b, d, d); }
inline static nk_f32_t sqeuclidean_f32(nk_f32_t const* a, nk_f32_t const* b, nk_size_t d) { return nk_metric_punned(nk_sqeuclidean_k, nk_f32_k, nk_cap_any_k)(a, b, d, d); }
*/
import "C"

// AngularI8 computes the cosine distance between two i8 vectors using the most suitable SIMD instruction set available.
func AngularI8(a, b []int8) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}

	return float32(C.angular_i8((*C.nk_i8_t)(&a[0]), (*C.nk_i8_t)(&b[0]), C.nk_size_t(len(a))))
}

// AngularF32 computes the cosine distance between two f32 vectors using the most suitable SIMD instruction set available.
func AngularF32(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}

	return float32(C.angular_f32((*C.nk_f32_t)(&a[0]), (*C.nk_f32_t)(&b[0]), C.nk_size_t(len(a))))
}

// InnerI8 computes the inner-product similarity between two i8 vectors using the most suitable SIMD instruction set available.
func InnerI8(a, b []int8) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}

	return float32(C.inner_i8((*C.nk_i8_t)(&a[0]), (*C.nk_i8_t)(&b[0]), C.nk_size_t(len(a))))
}

// InnerF32 computes the inner-product similarity between two f32 vectors using the most suitable SIMD instruction set available.
func InnerF32(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}

	return float32(C.inner_f32((*C.nk_f32_t)(&a[0]), (*C.nk_f32_t)(&b[0]), C.nk_size_t(len(a))))
}

// SqEuclideanI8 computes the squared Euclidean distance between two i8 vectors using the most suitable SIMD instruction set available.
func SqEuclideanI8(a, b []int8) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}

	return float32(C.sqeuclidean_i8((*C.nk_i8_t)(&a[0]), (*C.nk_i8_t)(&b[0]), C.nk_size_t(len(a))))
}

// SqEuclideanF32 computes the squared Euclidean distance between two f32 vectors using the most suitable SIMD instruction set available.
func SqEuclideanF32(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}

	return float32(C.sqeuclidean_f32((*C.nk_f32_t)(&a[0]), (*C.nk_f32_t)(&b[0]), C.nk_size_t(len(a))))
}
