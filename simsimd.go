package simsimd

import (
	"errors"
	"unsafe"
)

/*
#cgo LDFLAGS: -L.
#include "include/simsimd/simsimd.h"
#include <stdlib.h>

inline static simsimd_f32_t cosine_i8(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) { return simsimd_metric_punned(simsimd_metric_cosine_k, simsimd_datatype_i8_k, simsimd_cap_any_k)(a, b, d, d); }
inline static simsimd_f32_t cosine_f32(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t d) { return simsimd_metric_punned(simsimd_metric_cosine_k, simsimd_datatype_f32_k, simsimd_cap_any_k)(a, b, d, d); }

*/
import "C"


// CosineI8 computes the cosine similarity between two i8 vectors using the most suitable SIMD instruction set available.
func CosineI8(a, b []int8, d int) float32 {
	if len(a) != len(b) { panic(errors.New("vectors must have the same length")) }
	return float32(C.cosine_i8((*C.simsimd_i8_t)(&a[0]), (*C.simsimd_i8_t)(&b[0]), C.simsimd_size_t(d)))
}


// CosineF32 computes the cosine similarity between two i8 vectors using the most suitable SIMD instruction set available.
func CosineF32(a, b []float32, d int) float32 {
	if len(a) != len(b) { panic(errors.New("vectors must have the same length")) }
	return float32(C.cosine_f32((*C.simsimd_f32_t)(&a[0]), (*C.simsimd_f32_t)(&b[0]), C.simsimd_size_t(d)))
}

