package numkong

import (
	"math"
	"testing"
)

// region Dot Product Tests

func TestDotF64(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	result := DotF64(a, b)
	expected := float64(32) // 1*4 + 2*5 + 3*6 = 32
	if math.Abs(result-expected) > 1e-6 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestDotF32(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	result := DotF32(a, b)
	expected := float32(32)
	if math.Abs(float64(result-expected)) > 1e-3 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestDotI8(t *testing.T) {
	a := []int8{1, 2, 3}
	b := []int8{4, 5, 6}
	result := DotI8(a, b)
	expected := int32(32)
	if result != expected {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

// region Angular (Cosine) Distance Tests

func TestAngularI8(t *testing.T) {
	a := []int8{1, 0}
	b := []int8{0, 1}

	result := AngularI8(a, b)
	expected := float32(1.0) // Angular distance of orthogonal vectors is 1
	if math.Abs(float64(result-expected)) > 1e-3 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestAngularF32(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{0, 1}

	result := AngularF32(a, b)
	expected := float32(1.0) // Angular distance of orthogonal vectors is 1
	if math.Abs(float64(result-expected)) > 1e-3 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestAngularF64(t *testing.T) {
	a := []float64{1, 0}
	b := []float64{0, 1}

	result := AngularF64(a, b)
	expected := float64(1.0)
	if math.Abs(result-expected) > 1e-6 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestAngularIdentical(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{1, 2, 3}

	result := AngularF32(a, b)
	// Cosine distance of identical vectors is 0
	if math.Abs(float64(result)) > 0.01 {
		t.Errorf("Expected ~0, got %v", result)
	}
}

// region Euclidean Distance Tests

func TestEuclideanF64(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	result := EuclideanF64(a, b)
	// sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27) ≈ 5.196
	expected := math.Sqrt(27)
	if math.Abs(result-expected) > 0.01 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestEuclideanF32(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	result := EuclideanF32(a, b)
	expected := float32(math.Sqrt(27))
	if math.Abs(float64(result-expected)) > 0.01 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestEuclideanI8(t *testing.T) {
	a := []int8{1, 2, 3}
	b := []int8{4, 5, 6}
	result := EuclideanI8(a, b)
	expected := float32(math.Sqrt(27))
	if math.Abs(float64(result-expected)) > 0.01 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

// region Squared Euclidean Distance Tests

func TestSqEuclideanF64(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	result := SqEuclideanF64(a, b)
	expected := float64(27) // (4-1)^2 + (5-2)^2 + (6-3)^2 = 27
	if math.Abs(result-expected) > 1e-6 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestSqEuclideanF32(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	result := SqEuclideanF32(a, b)
	expected := float32(27)
	if math.Abs(float64(result-expected)) > 1e-3 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestSqEuclideanI8(t *testing.T) {
	a := []int8{1, 2, 3}
	b := []int8{4, 5, 6}
	result := SqEuclideanI8(a, b)
	expected := uint32(27)
	if result != expected {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

// region Geospatial Tests

func TestHaversineF64(t *testing.T) {
	// New York City to London
	// NYC: 40.7128° N, 74.0060° W
	// London: 51.5074° N, 0.1278° W
	degToRad := math.Pi / 180.0
	aLat := []float64{40.7128 * degToRad}
	aLon := []float64{-74.0060 * degToRad}
	bLat := []float64{51.5074 * degToRad}
	bLon := []float64{-0.1278 * degToRad}
	result := make([]float64, 1)

	success := HaversineF64(aLat, aLon, bLat, bLon, result)
	if !success {
		t.Error("HaversineF64 returned false")
	}
	// Expected: ~5,539 km
	expected := 5539000.0
	if math.Abs(result[0]-expected) > 5000 {
		t.Errorf("Expected ~%v m, got %v m", expected, result[0])
	}
}

func TestHaversineF32(t *testing.T) {
	degToRad := float32(math.Pi / 180.0)
	aLat := []float32{40.7128 * degToRad}
	aLon := []float32{-74.0060 * degToRad}
	bLat := []float32{51.5074 * degToRad}
	bLon := []float32{-0.1278 * degToRad}
	result := make([]float32, 1)

	success := HaversineF32(aLat, aLon, bLat, bLon, result)
	if !success {
		t.Error("HaversineF32 returned false")
	}
	expected := float32(5539000)
	if math.Abs(float64(result[0]-expected)) > 5000 {
		t.Errorf("Expected ~%v m, got %v m", expected, result[0])
	}
}

func TestVincentyF64(t *testing.T) {
	degToRad := math.Pi / 180.0
	aLat := []float64{40.7128 * degToRad}
	aLon := []float64{-74.0060 * degToRad}
	bLat := []float64{51.5074 * degToRad}
	bLon := []float64{-0.1278 * degToRad}
	result := make([]float64, 1)

	success := VincentyF64(aLat, aLon, bLat, bLon, result)
	if !success {
		t.Error("VincentyF64 returned false")
	}
	// Vincenty is more accurate, expected: ~5,570 km
	expected := 5570000.0
	if math.Abs(result[0]-expected) > 20000 {
		t.Errorf("Expected ~%v m, got %v m", expected, result[0])
	}
}

func TestVincentyF32(t *testing.T) {
	degToRad := float32(math.Pi / 180.0)
	aLat := []float32{40.7128 * degToRad}
	aLon := []float32{-74.0060 * degToRad}
	bLat := []float32{51.5074 * degToRad}
	bLon := []float32{-0.1278 * degToRad}
	result := make([]float32, 1)

	success := VincentyF32(aLat, aLon, bLat, bLon, result)
	if !success {
		t.Error("VincentyF32 returned false")
	}
	expected := float32(5570000)
	if math.Abs(float64(result[0]-expected)) > 20000 {
		t.Errorf("Expected ~%v m, got %v m", expected, result[0])
	}
}

// region Error Handling Tests

func TestVectorLengthMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()

	a := []int8{1, 0}
	b := []int8{0}
	_ = AngularI8(a, b) // This should panic
}

func TestEmptyVectors(t *testing.T) {
	a := []float32{}
	b := []float32{}
	result := DotF32(a, b)
	if result != 0 {
		t.Errorf("Expected 0 for empty vectors, got %v", result)
	}
}

func TestGeospatialLengthMismatch(t *testing.T) {
	aLat := []float64{1.0, 2.0}
	aLon := []float64{1.0}
	bLat := []float64{1.0}
	bLon := []float64{1.0}
	result := make([]float64, 1)

	success := HaversineF64(aLat, aLon, bLat, bLon, result)
	if success {
		t.Error("Expected false for mismatched lengths")
	}
}

// region Alias Tests

func TestAliases(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}

	// Test L2 aliases
	if L2F32(a, b) != EuclideanF32(a, b) {
		t.Error("L2F32 should equal EuclideanF32")
	}

	// Test L2sq aliases
	if L2sqF32(a, b) != SqEuclideanF32(a, b) {
		t.Error("L2sqF32 should equal SqEuclideanF32")
	}
}
