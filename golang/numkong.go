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

// CPU capability bit masks in chronological order (by first commercial silicon)
const (
	CapSerial      uint64 = 1 << 0  // Always: Fallback
	CapNeon        uint64 = 1 << 1  // 2013: ARM NEON
	CapHaswell     uint64 = 1 << 2  // 2013: Intel AVX2
	CapSkylake     uint64 = 1 << 3  // 2017: Intel AVX-512
	CapNeonHalf    uint64 = 1 << 4  // 2017: ARM NEON FP16
	CapNeonSdot    uint64 = 1 << 5  // 2017: ARM NEON i8 dot
	CapNeonFhm     uint64 = 1 << 6  // 2018: ARM NEON FP16 FML
	CapIcelake     uint64 = 1 << 7  // 2019: Intel AVX-512 VNNI
	CapGenoa       uint64 = 1 << 8  // 2020: AMD AVX-512 BF16
	CapNeonBfDot   uint64 = 1 << 9  // 2020: ARM NEON BF16
	CapSve         uint64 = 1 << 10 // 2020: ARM SVE
	CapSveHalf     uint64 = 1 << 11 // 2020: ARM SVE FP16
	CapSveSdot     uint64 = 1 << 12 // 2020: ARM SVE i8 dot
	CapSierra      uint64 = 1 << 13 // 2021: Intel AVX2+VNNI
	CapSveBfDot    uint64 = 1 << 14 // 2021: ARM SVE BF16
	CapSve2        uint64 = 1 << 15 // 2022: ARM SVE2
	CapV128Relaxed uint64 = 1 << 16 // 2022: WASM Relaxed SIMD
	CapSapphire    uint64 = 1 << 17 // 2023: Intel AVX-512 FP16
	CapSapphireAmx uint64 = 1 << 18 // 2023: Intel Sapphire AMX
	CapRvv         uint64 = 1 << 19 // 2023: RISC-V Vector
	CapRvvHalf     uint64 = 1 << 20 // 2023: RISC-V Zvfh
	CapRvvBf16     uint64 = 1 << 21 // 2023: RISC-V Zvfbfwma
	CapGraniteAmx  uint64 = 1 << 22 // 2024: Intel Granite AMX FP16
	CapTurin       uint64 = 1 << 23 // 2024: AMD Turin AVX-512 CD
	CapSme         uint64 = 1 << 24 // 2024: ARM SME
	CapSme2        uint64 = 1 << 25 // 2024: ARM SME2
	CapSmeF64      uint64 = 1 << 26 // 2024: ARM SME F64
	CapSmeFa64     uint64 = 1 << 27 // 2024: ARM SME FA64
	CapSve2p1      uint64 = 1 << 28 // 2025+: ARM SVE2.1
	CapSme2p1      uint64 = 1 << 29 // 2025+: ARM SME2.1
	CapSmeHalf     uint64 = 1 << 30 // 2025+: ARM SME F16F16
	CapSmeBf16     uint64 = 1 << 31 // 2025+: ARM SME B16B16
	CapSmeLut2     uint64 = 1 << 32 // 2025+: ARM SME LUTv2
)

// region Dot Product

// DotF64 computes the inner product (dot product) of two float64 vectors.
func DotF64(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f64_t
	C.nk_dot_f64((*C.nk_f64_t)(&a[0]), (*C.nk_f64_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float64(result)
}

// DotF32 computes the inner product (dot product) of two float32 vectors.
func DotF32(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f32_t
	C.nk_dot_f32((*C.nk_f32_t)(&a[0]), (*C.nk_f32_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float32(result)
}

// DotI8 computes the inner product (dot product) of two int8 vectors.
// Returns int32 (widened output).
func DotI8(a, b []int8) int32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_i32_t
	C.nk_dot_i8((*C.nk_i8_t)(&a[0]), (*C.nk_i8_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return int32(result)
}

// endregion

// region Angular (Cosine) Distance

// AngularF64 computes the angular (cosine) distance between two float64 vectors.
func AngularF64(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f64_t
	C.nk_angular_f64((*C.nk_f64_t)(&a[0]), (*C.nk_f64_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float64(result)
}

// AngularF32 computes the angular (cosine) distance between two float32 vectors.
func AngularF32(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f32_t
	C.nk_angular_f32((*C.nk_f32_t)(&a[0]), (*C.nk_f32_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float32(result)
}

// AngularI8 computes the angular (cosine) distance between two int8 vectors.
// Returns float32 (widened output).
func AngularI8(a, b []int8) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f32_t
	C.nk_angular_i8((*C.nk_i8_t)(&a[0]), (*C.nk_i8_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float32(result)
}

// endregion

// region Euclidean Distance (L2)

// EuclideanF64 computes the Euclidean (L2) distance between two float64 vectors.
func EuclideanF64(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f64_t
	C.nk_l2_f64((*C.nk_f64_t)(&a[0]), (*C.nk_f64_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float64(result)
}

// EuclideanF32 computes the Euclidean (L2) distance between two float32 vectors.
func EuclideanF32(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f32_t
	C.nk_l2_f32((*C.nk_f32_t)(&a[0]), (*C.nk_f32_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float32(result)
}

// EuclideanI8 computes the Euclidean (L2) distance between two int8 vectors.
// Returns float32 (widened output).
func EuclideanI8(a, b []int8) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f32_t
	C.nk_l2_i8((*C.nk_i8_t)(&a[0]), (*C.nk_i8_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float32(result)
}

// L2F64 is an alias for EuclideanF64.
func L2F64(a, b []float64) float64 { return EuclideanF64(a, b) }

// L2F32 is an alias for EuclideanF32.
func L2F32(a, b []float32) float32 { return EuclideanF32(a, b) }

// L2I8 is an alias for EuclideanI8.
func L2I8(a, b []int8) float32 { return EuclideanI8(a, b) }

// endregion

// region Squared Euclidean Distance (L2sq)

// SqEuclideanF64 computes the squared Euclidean distance between two float64 vectors.
func SqEuclideanF64(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f64_t
	C.nk_l2sq_f64((*C.nk_f64_t)(&a[0]), (*C.nk_f64_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float64(result)
}

// SqEuclideanF32 computes the squared Euclidean distance between two float32 vectors.
func SqEuclideanF32(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f32_t
	C.nk_l2sq_f32((*C.nk_f32_t)(&a[0]), (*C.nk_f32_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float32(result)
}

// SqEuclideanI8 computes the squared Euclidean distance between two int8 vectors.
// Returns uint32 (widened output).
func SqEuclideanI8(a, b []int8) uint32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_u32_t
	C.nk_l2sq_i8((*C.nk_i8_t)(&a[0]), (*C.nk_i8_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return uint32(result)
}

// L2sqF64 is an alias for SqEuclideanF64.
func L2sqF64(a, b []float64) float64 { return SqEuclideanF64(a, b) }

// L2sqF32 is an alias for SqEuclideanF32.
func L2sqF32(a, b []float32) float32 { return SqEuclideanF32(a, b) }

// L2sqI8 is an alias for SqEuclideanI8.
func L2sqI8(a, b []int8) uint32 { return SqEuclideanI8(a, b) }

// endregion

// region Geospatial Distance

// HaversineF64 computes the Haversine (great-circle) distance between coordinate pairs.
// Input coordinates must be in radians. Output distances are in meters.
// aLat, aLon: latitudes and longitudes of first set of points
// bLat, bLon: latitudes and longitudes of second set of points
// result: output slice for distances (must be same length as input slices)
func HaversineF64(aLat, aLon, bLat, bLon, result []float64) bool {
	n := len(aLat)
	if n == 0 || n != len(aLon) || n != len(bLat) || n != len(bLon) || n != len(result) {
		return false
	}
	C.nk_haversine_f64(
		(*C.nk_f64_t)(&aLat[0]), (*C.nk_f64_t)(&aLon[0]),
		(*C.nk_f64_t)(&bLat[0]), (*C.nk_f64_t)(&bLon[0]),
		C.nk_size_t(n), (*C.nk_f64_t)(&result[0]))
	return true
}

// HaversineF32 computes the Haversine (great-circle) distance between coordinate pairs.
// Input coordinates must be in radians. Output distances are in meters.
func HaversineF32(aLat, aLon, bLat, bLon, result []float32) bool {
	n := len(aLat)
	if n == 0 || n != len(aLon) || n != len(bLat) || n != len(bLon) || n != len(result) {
		return false
	}
	C.nk_haversine_f32(
		(*C.nk_f32_t)(&aLat[0]), (*C.nk_f32_t)(&aLon[0]),
		(*C.nk_f32_t)(&bLat[0]), (*C.nk_f32_t)(&bLon[0]),
		C.nk_size_t(n), (*C.nk_f32_t)(&result[0]))
	return true
}

// VincentyF64 computes the Vincenty (ellipsoidal geodesic) distance between coordinate pairs.
// Input coordinates must be in radians. Output distances are in meters.
// More accurate than Haversine for long distances.
func VincentyF64(aLat, aLon, bLat, bLon, result []float64) bool {
	n := len(aLat)
	if n == 0 || n != len(aLon) || n != len(bLat) || n != len(bLon) || n != len(result) {
		return false
	}
	C.nk_vincenty_f64(
		(*C.nk_f64_t)(&aLat[0]), (*C.nk_f64_t)(&aLon[0]),
		(*C.nk_f64_t)(&bLat[0]), (*C.nk_f64_t)(&bLon[0]),
		C.nk_size_t(n), (*C.nk_f64_t)(&result[0]))
	return true
}

// VincentyF32 computes the Vincenty (ellipsoidal geodesic) distance between coordinate pairs.
// Input coordinates must be in radians. Output distances are in meters.
func VincentyF32(aLat, aLon, bLat, bLon, result []float32) bool {
	n := len(aLat)
	if n == 0 || n != len(aLon) || n != len(bLat) || n != len(bLon) || n != len(result) {
		return false
	}
	C.nk_vincenty_f32(
		(*C.nk_f32_t)(&aLat[0]), (*C.nk_f32_t)(&aLon[0]),
		(*C.nk_f32_t)(&bLat[0]), (*C.nk_f32_t)(&bLon[0]),
		C.nk_size_t(n), (*C.nk_f32_t)(&result[0]))
	return true
}

// endregion

// Ensure unsafe is used (for CGO pointer conversions)
var _ = unsafe.Pointer(nil)
