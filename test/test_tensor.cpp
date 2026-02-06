/**
 *  @brief C++ vector type instantiation tests.
 *  @file test/test_tensor.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */
#include <cassert>
#include <complex>

#include "test.hpp"

void test_vector_types() {
    std::printf("Testing vector type instantiations...\n");

    // Test: float (primitive, 1 dim per value)
    {
        nk::vector<float> v;
        assert(v.resize(100) && "float resize failed");
        assert(v.size() == 100 && "float size mismatch");
        assert(v.size_values() == 100 && "float size_values mismatch");
        v[50] = 3.14f;
        assert(v[50] == 3.14f && "float operator[] failed");
        std::size_t count = 0;
        for (auto it = v.begin(); it != v.end(); ++it) ++count;
        assert(count == 100 && "float iterator count mismatch");
    }

    // Test: std::complex<double> (primitive complex, 1 dim per value)
    {
        nk::vector<std::complex<double>> v;
        assert(v.resize(50) && "complex<double> resize failed");
        assert(v.size() == 50 && "complex<double> size mismatch");
        assert(v.size_values() == 50 && "complex<double> size_values mismatch");
        v[25] = std::complex<double>(1.0, 2.0);
        assert(v[25] == std::complex<double>(1.0, 2.0) && "complex<double> operator[] failed");
        std::size_t count = 0;
        for (auto &elem : v) {
            (void)elem;
            ++count;
        }
        assert(count == 50 && "complex<double> range-for count mismatch");
    }

    // Test: f32c_t (NumKong complex wrapper, 1 dim per value)
    {
        nk::vector<nk::f32c_t> v;
        assert(v.resize(64) && "f32c_t resize failed");
        assert(v.size() == 64 && "f32c_t size mismatch");
        assert(v.size_values() == 64 && "f32c_t size_values mismatch");
        v[32] = nk::f32c_t(1.5f, -2.5f);
        assert(v[32] == nk::f32c_t(1.5f, -2.5f) && "f32c_t operator[] failed");
    }

    // Test: i8_t (NumKong integer wrapper, 1 dim per value)
    {
        nk::vector<nk::i8_t> v;
        assert(v.resize(128) && "i8_t resize failed");
        assert(v.size() == 128 && "i8_t size mismatch");
        assert(v.size_values() == 128 && "i8_t size_values mismatch");
        v[64] = nk::i8_t(-42);
        assert(v[64] == nk::i8_t(-42) && "i8_t operator[] failed");
    }

    // Test: i4x2_t (sub-byte, 2 dims per value, LSB-first)
    {
        nk::vector<nk::i4x2_t> v;
        assert(v.resize(100) && "i4x2_t resize failed");
        assert(v.size() == 100 && "i4x2_t size mismatch");
        assert(v.size_values() == 50 && "i4x2_t size_values mismatch (should be dims/2)");

        v[0] = 5, v[1] = -3;
        assert(v[0] == 5 && "i4x2_t dim 0 mismatch");
        assert(v[1] == -3 && "i4x2_t dim 1 mismatch");

        // Test iterator returns correct count
        std::size_t count = 0;
        for (auto it = v.begin(); it != v.end(); ++it) ++count;
        assert(count == 100 && "i4x2_t iterator count mismatch");
    }

    // Test: u1x8_t (sub-byte, 8 dims per value, LSB-first)
    {
        nk::vector<nk::u1x8_t> v;
        assert(v.resize(64) && "u1x8_t resize failed");
        assert(v.size() == 64 && "u1x8_t size mismatch");
        assert(v.size_values() == 8 && "u1x8_t size_values mismatch (should be dims/8)");

        v[0] = true, v[1] = false, v[7] = true;
        assert(v[0] == true && "u1x8_t dim 0 mismatch");
        assert(v[1] == false && "u1x8_t dim 1 mismatch");
        assert(v[7] == true && "u1x8_t dim 7 mismatch");

        // Test iterator returns correct count
        std::size_t count = 0;
        for (auto it = v.begin(); it != v.end(); ++it) ++count;
        assert(count == 64 && "u1x8_t iterator count mismatch");
    }

    // Test: Custom allocator (stateless)
    {
        using custom_alloc_t = nk::aligned_allocator<nk::f32_t, 128>;
        nk::vector<nk::f32_t, custom_alloc_t> v;
        assert(v.resize(256) && "custom allocator resize failed");
        assert(v.size() == 256 && "custom allocator size mismatch");
        v[128] = nk::f32_t(99.0f);
        assert(v[128] == nk::f32_t(99.0f) && "custom allocator value mismatch");
    }

    // Test: Reserve and capacity
    {
        nk::vector<nk::f64_t> v;
        assert(v.reserve(1000) && "reserve failed");
        assert(v.capacity() >= 1000 && "capacity < reserved");
        assert(v.resize(500) && "resize after reserve failed");
        assert(v.size() == 500 && "size after reserve mismatch");
        assert(v.capacity() >= 1000 && "capacity shrunk after resize");
    }

    // Test: Move semantics
    {
        nk::vector<nk::f32_t> v1;
        assert(v1.resize(100) && "v1 resize failed");
        v1[50] = nk::f32_t(42.0f);

        nk::vector<nk::f32_t> v2 = std::move(v1);
        assert(v2.size() == 100 && "move ctor size mismatch");
        assert(v2[50] == nk::f32_t(42.0f) && "move ctor value mismatch");
        assert(v1.size() == 0 && "moved-from vector not empty");

        nk::vector<nk::f32_t> v3;
        v3 = std::move(v2);
        assert(v3.size() == 100 && "move assign size mismatch");
        assert(v3[50] == nk::f32_t(42.0f) && "move assign value mismatch");
    }

    // Test: Swap
    {
        nk::vector<nk::i8_t> v1, v2;
        assert(v1.resize(10) && v2.resize(20));
        v1[0] = nk::i8_t(1);
        v2[0] = nk::i8_t(2);

        swap(v1, v2);
        assert(v1.size() == 20 && "swap v1 size mismatch");
        assert(v2.size() == 10 && "swap v2 size mismatch");
        assert(v1[0] == nk::i8_t(2) && "swap v1 value mismatch");
        assert(v2[0] == nk::i8_t(1) && "swap v2 value mismatch");
    }

    std::printf("  vector<float>:                OK\n");
    std::printf("  vector<std::complex<double>>: OK\n");
    std::printf("  vector<f32c_t>:               OK\n");
    std::printf("  vector<i8_t>:                 OK\n");
    std::printf("  vector<i4x2_t>:               OK (sub-byte proxy, LSB-first)\n");
    std::printf("  vector<u1x8_t>:               OK (sub-byte proxy, LSB-first)\n");
    std::printf("  custom allocator:             OK\n");
    std::printf("  reserve/capacity:             OK\n");
    std::printf("  move semantics:               OK\n");
    std::printf("  swap:                         OK\n");
}
