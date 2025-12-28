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
