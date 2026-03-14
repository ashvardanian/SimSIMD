package numkong

/*
#cgo CFLAGS: -O3 -I../include
#cgo LDFLAGS: -O3 -L. -lm
#define NK_NATIVE_F16 (0)
#define NK_NATIVE_BF16 (0)
#include "numkong/numkong.h"
*/
import "C"

// HaversineF64 computes the Haversine (great-circle) distance between coordinate pairs.
// Input coordinates must be in radians. Output distances are in meters.
// All slices must have the same length.
func HaversineF64(aLat, aLon, bLat, bLon, result []float64) {
	n := len(aLat)
	if n != len(aLon) || n != len(bLat) || n != len(bLon) || n != len(result) {
		panic("all coordinate and result slices must have the same length")
	}
	if n == 0 {
		return
	}
	C.nk_haversine_f64(
		(*C.nk_f64_t)(&aLat[0]), (*C.nk_f64_t)(&aLon[0]),
		(*C.nk_f64_t)(&bLat[0]), (*C.nk_f64_t)(&bLon[0]),
		C.nk_size_t(n), (*C.nk_f64_t)(&result[0]))
}

// HaversineF32 computes the Haversine (great-circle) distance between coordinate pairs.
// Input coordinates must be in radians. Output distances are in meters.
// All slices must have the same length.
func HaversineF32(aLat, aLon, bLat, bLon, result []float32) {
	n := len(aLat)
	if n != len(aLon) || n != len(bLat) || n != len(bLon) || n != len(result) {
		panic("all coordinate and result slices must have the same length")
	}
	if n == 0 {
		return
	}
	C.nk_haversine_f32(
		(*C.nk_f32_t)(&aLat[0]), (*C.nk_f32_t)(&aLon[0]),
		(*C.nk_f32_t)(&bLat[0]), (*C.nk_f32_t)(&bLon[0]),
		C.nk_size_t(n), (*C.nk_f32_t)(&result[0]))
}

// VincentyF64 computes the Vincenty (ellipsoidal geodesic) distance between coordinate pairs.
// Input coordinates must be in radians. Output distances are in meters.
// More accurate than Haversine for long distances.
// All slices must have the same length.
func VincentyF64(aLat, aLon, bLat, bLon, result []float64) {
	n := len(aLat)
	if n != len(aLon) || n != len(bLat) || n != len(bLon) || n != len(result) {
		panic("all coordinate and result slices must have the same length")
	}
	if n == 0 {
		return
	}
	C.nk_vincenty_f64(
		(*C.nk_f64_t)(&aLat[0]), (*C.nk_f64_t)(&aLon[0]),
		(*C.nk_f64_t)(&bLat[0]), (*C.nk_f64_t)(&bLon[0]),
		C.nk_size_t(n), (*C.nk_f64_t)(&result[0]))
}

// VincentyF32 computes the Vincenty (ellipsoidal geodesic) distance between coordinate pairs.
// Input coordinates must be in radians. Output distances are in meters.
// All slices must have the same length.
func VincentyF32(aLat, aLon, bLat, bLon, result []float32) {
	n := len(aLat)
	if n != len(aLon) || n != len(bLat) || n != len(bLon) || n != len(result) {
		panic("all coordinate and result slices must have the same length")
	}
	if n == 0 {
		return
	}
	C.nk_vincenty_f32(
		(*C.nk_f32_t)(&aLat[0]), (*C.nk_f32_t)(&aLon[0]),
		(*C.nk_f32_t)(&bLat[0]), (*C.nk_f32_t)(&bLon[0]),
		C.nk_size_t(n), (*C.nk_f32_t)(&result[0]))
}
