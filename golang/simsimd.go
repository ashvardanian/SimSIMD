package simsimd

import (
	"errors"
	"unsafe"
)

/*
#cgo LDFLAGS: -L.
#include "../include/simsimd/simsimd.h"
#include <stdlib.h>

inline static simsimd_f32_t simsimd_default_i8_cos(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
#if SIMSIMD_TARGET_ARM_NEON
    return simsimd_neon_i8_cos(a, b, d);
#elif SIMSIMD_TARGET_X86_AVX2
    return simsimd_avx2_i8_cos(a, b, d);
#elif SIMSIMD_TARGET_X86_AVX512
    return simsimd_avx512_i8_cos(a, b, d);
#else
    return simsimd_auto_i8_cos(a, b, d);
#endif
}

*/
import "C"


// CosineI8 computes the cosine similarity between two i8 vectors using the most suitable SIMD instruction set available.
func CosineI8(a, b []int8, d int) float32 {
	if len(a) != len(b) { panic(errors.New("vectors must have the same length")) }
	return float32(C.simsimd_default_i8_cos((*C.simsimd_i8_t)(&a[0]), (*C.simsimd_i8_t)(&b[0]), C.simsimd_size_t(d)))
}

