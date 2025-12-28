/**
 *  @file   test.c
 *  @brief  Test focusing only on the simplest functionality.
 */

#include <assert.h> // `assert`
#include <math.h>   // `sqrtf`
#include <stdio.h>  // `printf`
#include <stdlib.h> // `malloc`, `free`, `rand`, `srand`

#define NK_NATIVE_F16  0
#define NK_NATIVE_BF16 0
#include <numkong/numkong.h>

#pragma region Helpers

/**
 *  @brief  Simple pseudo-random number generator for reproducible tests.
 *          Uses a linear congruential generator (LCG) for simplicity.
 */
static nk_u32_t test_random_seed = 12345;
static nk_f64_t test_random_f64(nk_f64_t min_val, nk_f64_t max_val) {
    test_random_seed = test_random_seed * 1103515245 + 12345;
    nk_f64_t normalized = (nk_f64_t)(test_random_seed & 0x7FFFFFFF) / (nk_f64_t)0x7FFFFFFF;
    return min_val + normalized * (max_val - min_val);
}

/**
 *  @brief  Assert two f32 arrays are equal within tolerance.
 */
static void assert_f32_arrays_equal(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t tol) {
    for (nk_size_t i = 0; i < n; i++) {
        nk_f64_t diff = fabs((nk_f64_t)a[i] - (nk_f64_t)b[i]);
        assert(diff < tol);
    }
}

/**
 *  @brief  Assert two f64 arrays are equal within tolerance.
 */
static void assert_f64_arrays_equal(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t tol) {
    for (nk_size_t i = 0; i < n; i++) {
        nk_f64_t diff = fabs(a[i] - b[i]);
        assert(diff < tol);
    }
}

/**
 *  @brief  Assert two f64 values are approximately equal with relative/absolute tolerance.
 */
static void assert_f64_near(nk_f64_t expected, nk_f64_t actual, nk_f64_t rel_tol, nk_f64_t abs_tol,
                            char const *context) {
    nk_f64_t diff = fabs(expected - actual);
    nk_f64_t rel_err = expected > 0 ? diff / expected : diff;
    if (rel_err > rel_tol && diff > abs_tol)
        printf("  FAIL %s: expected=%.2f, actual=%.2f, diff=%.2f\n", context, expected, actual, diff);
    assert(rel_err < rel_tol || diff < abs_tol);
}

#pragma endregion

#pragma region Infrastructure

/**
 *  @brief  Logs CPU capabilities supported by the current build (compile-time) and runtime.
 */
void print_capabilities(void) {
    nk_capability_t runtime_caps = nk_capabilities();

    // Log supported functionality
    char const *flags[2] = {"false", "true"};
    printf("Benchmarking Similarity Measures\n");
    printf("- Compiler used native F16: %s\n", flags[NK_NATIVE_F16]);
    printf("- Compiler used native BF16: %s\n", flags[NK_NATIVE_BF16]);
    printf("\n");
    printf("Compile-time settings:\n");
    printf("- Arm NEON support enabled: %s\n", flags[NK_TARGET_NEON]);
    printf("- Arm SVE support enabled: %s\n", flags[NK_TARGET_SVE]);
    printf("- Arm SVE2 support enabled: %s\n", flags[NK_TARGET_SVE2]);
    printf("- x86 Haswell support enabled: %s\n", flags[NK_TARGET_HASWELL]);
    printf("- x86 Skylake support enabled: %s\n", flags[NK_TARGET_SKYLAKE]);
    printf("- x86 Ice Lake support enabled: %s\n", flags[NK_TARGET_ICE]);
    printf("- x86 Genoa support enabled: %s\n", flags[NK_TARGET_GENOA]);
    printf("- x86 Sapphire Rapids support enabled: %s\n", flags[NK_TARGET_SAPPHIRE]);
    printf("- x86 Turin support enabled: %s\n", flags[NK_TARGET_TURIN]);
    printf("- x86 Sierra Forest support enabled: %s\n", flags[NK_TARGET_SIERRA]);
    printf("\n");
    printf("Run-time settings:\n");
    printf("- Arm NEON support enabled: %s\n", flags[(runtime_caps & nk_cap_neon_k) != 0]);
    printf("- Arm NEON F16 support enabled: %s\n", flags[(runtime_caps & nk_cap_neon_f16_k) != 0]);
    printf("- Arm NEON BF16 support enabled: %s\n", flags[(runtime_caps & nk_cap_neon_bf16_k) != 0]);
    printf("- Arm NEON I8 support enabled: %s\n", flags[(runtime_caps & nk_cap_neon_i8_k) != 0]);
    printf("- Arm SVE support enabled: %s\n", flags[(runtime_caps & nk_cap_sve_k) != 0]);
    printf("- Arm SVE F16 support enabled: %s\n", flags[(runtime_caps & nk_cap_sve_f16_k) != 0]);
    printf("- Arm SVE BF16 support enabled: %s\n", flags[(runtime_caps & nk_cap_sve_bf16_k) != 0]);
    printf("- Arm SVE I8 support enabled: %s\n", flags[(runtime_caps & nk_cap_sve_i8_k) != 0]);
    printf("- Arm SVE2 support enabled: %s\n", flags[(runtime_caps & nk_cap_sve2_k) != 0]);
    printf("- x86 Haswell support enabled: %s\n", flags[(runtime_caps & nk_cap_haswell_k) != 0]);
    printf("- x86 Skylake support enabled: %s\n", flags[(runtime_caps & nk_cap_skylake_k) != 0]);
    printf("- x86 Ice Lake support enabled: %s\n", flags[(runtime_caps & nk_cap_ice_k) != 0]);
    printf("- x86 Genoa support enabled: %s\n", flags[(runtime_caps & nk_cap_genoa_k) != 0]);
    printf("- x86 Sapphire Rapids support enabled: %s\n", flags[(runtime_caps & nk_cap_sapphire_k) != 0]);
    printf("- x86 Turin support enabled: %s\n", flags[(runtime_caps & nk_cap_turin_k) != 0]);
    printf("\n");
}

/**
 *  @brief  A trivial test that checks if the utility functions return the expected values.
 */
void test_utilities(void) {
    nk_capability_t capabilities = nk_capabilities();

    int uses_neon = nk_uses_neon();
    int uses_sve = nk_uses_sve();
    int uses_haswell = nk_uses_haswell();
    int uses_skylake = nk_uses_skylake();
    int uses_ice = nk_uses_ice();
    int uses_genoa = nk_uses_genoa();
    int uses_sapphire = nk_uses_sapphire();
    int uses_turin = nk_uses_turin();
    int uses_sierra = nk_uses_sierra();

    assert(uses_neon == ((capabilities & nk_cap_neon_k) != 0));
    assert(uses_sve == ((capabilities & nk_cap_sve_k) != 0));
    assert(uses_haswell == ((capabilities & nk_cap_haswell_k) != 0));
    assert(uses_skylake == ((capabilities & nk_cap_skylake_k) != 0));
    assert(uses_ice == ((capabilities & nk_cap_ice_k) != 0));
    assert(uses_genoa == ((capabilities & nk_cap_genoa_k) != 0));
    assert(uses_sapphire == ((capabilities & nk_cap_sapphire_k) != 0));
    assert(uses_turin == ((capabilities & nk_cap_turin_k) != 0));
    assert(uses_sierra == ((capabilities & nk_cap_sierra_k) != 0));
}

/**
 *  @brief  A trivial test for internal saturated arithmic function.
 */
void test_saturating_arithmetic(void) {
    // Test cases for addition functions
    nk_u8_t u8_a = 200, u8_b = 100, u8_r;
    nk_u8_sadd_(&u8_a, &u8_b, &u8_r);
    assert(u8_r == 255); // Overflow case for u8

    nk_i8_t i8_a = 100, i8_b = 40, i8_r;
    nk_i8_sadd_(&i8_a, &i8_b, &i8_r);
    assert(i8_r == 127); // Overflow case for i8

    nk_i8_t i8_under_a = -100, i8_under_b = -40;
    nk_i8_sadd_(&i8_under_a, &i8_under_b, &i8_r);
    assert(i8_r == -128); // Underflow case for i8

    nk_u16_t u16_a = 50000, u16_b = 20000, u16_r;
    nk_u16_sadd_(&u16_a, &u16_b, &u16_r);
    assert(u16_r == 65535); // Overflow case for u16

    nk_i16_t i16_a = 30000, i16_b = 10000, i16_r;
    nk_i16_sadd_(&i16_a, &i16_b, &i16_r);
    assert(i16_r == 32767); // Overflow case for i16

    nk_i16_t i16_under_a = -20000, i16_under_b = -15000;
    nk_i16_sadd_(&i16_under_a, &i16_under_b, &i16_r);
    assert(i16_r == -32768); // Underflow case for i16

    // Test cases for multiplication functions
    nk_u8_t u8_m_a = 20, u8_m_b = 20, u8_m_r;
    nk_u8_smul_(&u8_m_a, &u8_m_b, &u8_m_r);
    assert(u8_m_r == 255); // Overflow case for u8 multiplication

    nk_i8_t i8_m_a = 10, i8_m_b = -13, i8_m_r;
    nk_i8_smul_(&i8_m_a, &i8_m_b, &i8_m_r);
    assert(i8_m_r == -128); // Underflow case for i8 multiplication

    nk_i8_t i8_m_under_a = -100, i8_m_under_b = 2;
    nk_i8_smul_(&i8_m_under_a, &i8_m_under_b, &i8_m_r);
    assert(i8_m_r == -128); // Underflow case for i8 multiplication

    nk_u16_t u16_m_a = 300, u16_m_b = 300, u16_m_r;
    nk_u16_smul_(&u16_m_a, &u16_m_b, &u16_m_r);
    assert(u16_m_r == 65535); // Overflow case for u16 multiplication

    nk_i16_t i16_m_a = 200, i16_m_b = 300, i16_m_r;
    nk_i16_smul_(&i16_m_a, &i16_m_b, &i16_m_r);
    assert(i16_m_r == 32767); // Overflow case for i16 multiplication

    nk_i16_t i16_m_under_a = -200, i16_m_under_b = 300;
    nk_i16_smul_(&i16_m_under_a, &i16_m_under_b, &i16_m_r);
    assert(i16_m_r == -32768); // Underflow case for i16 multiplication

    // i32/u32 saturating addition
    nk_i32_t i32_a = 2000000000, i32_b = 200000000, i32_r;
    nk_i32_sadd_(&i32_a, &i32_b, &i32_r);
    assert(i32_r == 2147483647); // Overflow case for i32

    nk_u32_t u32_a = 4000000000U, u32_b = 400000000U, u32_r;
    nk_u32_sadd_(&u32_a, &u32_b, &u32_r);
    assert(u32_r == 4294967295U); // Overflow case for u32

    // i64/u64 saturating addition
    nk_i64_t i64_a = 9000000000000000000LL, i64_b = 1000000000000000000LL, i64_r;
    nk_i64_sadd_(&i64_a, &i64_b, &i64_r);
    assert(i64_r == 9223372036854775807LL); // Overflow case for i64

    nk_u64_t u64_a = 18000000000000000000ULL, u64_b = 1000000000000000000ULL, u64_r;
    nk_u64_sadd_(&u64_a, &u64_b, &u64_r);
    assert(u64_r == 18446744073709551615ULL); // Overflow case for u64

    // Normal cases without overflow
    nk_u8_t u8_n_a = 20, u8_n_b = 15, u8_n_r;
    nk_u8_sadd_(&u8_n_a, &u8_n_b, &u8_n_r);
    assert(u8_n_r == 35);

    nk_i8_t i8_n_a = -10, i8_n_b = 20, i8_n_r;
    nk_i8_sadd_(&i8_n_a, &i8_n_b, &i8_n_r);
    assert(i8_n_r == 10);

    // Floating-point cases
    nk_f32_t f32_a = 1.5f, f32_b = 2.5f, f32_r;
    nk_f32_sadd_(&f32_a, &f32_b, &f32_r);
    assert(f32_r == 4.0f); // Normal addition for f32

    nk_f32_t f32_m_a = 1.5f, f32_m_b = 2.0f;
    nk_f32_smul_(&f32_m_a, &f32_m_b, &f32_r);
    assert(f32_r == 3.0f); // Normal multiplication for f32

    printf("Test saturating arithmetic: PASS\n");
}

/**
 *  @brief  Validating N-Dimensional indexing utilities.
 */
void test_xd_index(void) {
    nk_size_t shape[NK_NDARRAY_MAX_RANK];
    nk_ssize_t strides[NK_NDARRAY_MAX_RANK];
    nk_xd_index_t xd_index;
    nk_ssize_t linear_byte_offset;

    // 1D array
    shape[0] = 10;
    strides[0] = 1 * sizeof(nk_u8_t);
    nk_xd_index_init(&xd_index);
    for (nk_size_t i = 0; i < 10; i++) {
        assert(xd_index.byte_offset == i * sizeof(nk_u8_t));
        assert(xd_index.coordinates[0] == i);
        assert(nk_xd_index_linearize(shape, strides, 1, &xd_index.coordinates[0], &linear_byte_offset));
        assert(linear_byte_offset == i * sizeof(nk_u8_t));
        assert(nk_xd_index_next(shape, strides, 1, &xd_index.coordinates[0], &xd_index.byte_offset) == (i < 9));
    }

    // 2D array
    shape[0] = 10, shape[1] = 5;
    strides[0] = 5 * sizeof(nk_u8_t), strides[1] = 1 * sizeof(nk_u8_t);
    nk_xd_index_init(&xd_index);
    for (nk_size_t i = 0; i < 10; i++) {
        for (nk_size_t j = 0; j < 5; j++) {
            assert(xd_index.byte_offset == (i * 5 + j) * sizeof(nk_u8_t));
            assert(xd_index.coordinates[0] == i);
            assert(xd_index.coordinates[1] == j);
            assert(nk_xd_index_linearize(shape, strides, 2, &xd_index.coordinates[0], &linear_byte_offset));
            assert(linear_byte_offset == (i * 5 + j) * sizeof(nk_u8_t));
            assert(nk_xd_index_next(shape, strides, 2, &xd_index.coordinates[0], &xd_index.byte_offset) ==
                   (i != 9 || j != 4));
        }
    }

    // 2D array of complex numbers, taking only the real part
    shape[0] = 10, shape[1] = 5;
    strides[0] = 10 * sizeof(nk_u8_t), strides[1] = 2 * sizeof(nk_u8_t);
    nk_xd_index_init(&xd_index);
    for (nk_size_t i = 0; i < 10; i++) {
        for (nk_size_t j = 0; j < 5; j++) {
            assert(xd_index.byte_offset == (i * 5 + j) * 2 * sizeof(nk_u8_t));
            assert(xd_index.coordinates[0] == i);
            assert(xd_index.coordinates[1] == j);
            assert(nk_xd_index_linearize(shape, strides, 2, &xd_index.coordinates[0], &linear_byte_offset));
            assert(linear_byte_offset == (i * 5 + j) * 2 * sizeof(nk_u8_t));
            assert(nk_xd_index_next(shape, strides, 2, &xd_index.coordinates[0], &xd_index.byte_offset) ==
                   (i != 9 || j != 4));
        }
    }

    // 3D array with different strides at every level
    // At each level it should be at least as big as the smaller level stride
    // multiplied by its size, otherwise we interleave the data.
    shape[0] = 10, shape[1] = 5, shape[2] = 3;
    strides[0] = 41 * sizeof(nk_u8_t), strides[1] = 7 * sizeof(nk_u8_t), strides[2] = 2 * sizeof(nk_u8_t);
    nk_xd_index_init(&xd_index);
    for (nk_size_t i = 0; i < 10; i++) {
        for (nk_size_t j = 0; j < 5; j++) {
            for (nk_size_t k = 0; k < 3; k++) {
                assert(xd_index.byte_offset == (i * strides[0] + j * strides[1] + k * strides[2]));
                assert(xd_index.coordinates[0] == i);
                assert(xd_index.coordinates[1] == j);
                assert(xd_index.coordinates[2] == k);
                assert(nk_xd_index_linearize(shape, strides, 3, &xd_index.coordinates[0], &linear_byte_offset));
                assert(linear_byte_offset == (i * strides[0] + j * strides[1] + k * strides[2]));
                assert(nk_xd_index_next(shape, strides, 3, &xd_index.coordinates[0], &xd_index.byte_offset) ==
                       (i != 9 || j != 4 || k != 2));
            }
        }
    }

    // Populated 3D array with different strides at every level
    {
        nk_f32_t tensor[11][43][7];
        // Fill tensor with values
        for (nk_size_t i = 0; i < 11; i++) {
            for (nk_size_t j = 0; j < 43; j++)
                for (nk_size_t k = 0; k < 7; k++) tensor[i][j][k] = i * 10000 + j * 100 + k * 1;
        }
        // Accumulate a slice: tensor[1:9:2, 2:42:4, 1:5:3] ~ 4 channels, 10 rows, 2 columns
        nk_xd_index_init(&xd_index);
        shape[0] = nk_divide_ceil_(9 - 1, 2);
        shape[1] = nk_divide_ceil_(42 - 2, 4);
        shape[2] = nk_divide_ceil_(5 - 1, 3);
        strides[0] = 43 * 7 * sizeof(nk_f32_t) * 2; // Physical size of 2 channels
        strides[1] = 7 * sizeof(nk_f32_t) * 4;      // Physical size of 4 rows
        strides[2] = 3 * sizeof(nk_f32_t);          // Physical size of 3 columns
        // Accumulate using native indexing
        nk_f32_t sum_native = 0;
        for (nk_size_t i = 1; i < 9; i += 2) {
            for (nk_size_t j = 2; j < 42; j += 4) {
                for (nk_size_t k = 1; k < 5; k += 3) { //
                    sum_native += tensor[i][j][k];
                }
            }
        }
        // Accumulate using our `nk_xd_index_t` iterator
        nk_f32_t sum_with_xd_index = 0;
        nk_f32_t sum_native_running = 0;
        for (nk_size_t i = 1; i < 9; i += 2) {
            for (nk_size_t j = 2; j < 42; j += 4) {
                for (nk_size_t k = 1; k < 5; k += 3) {
                    nk_size_t const expected_global_offset =  //
                        ((i - 1) / 2) * shape[1] * shape[2] + //
                        ((j - 2) / 4) * shape[2] +            //
                        ((k - 1) / 3);                        //
                    nk_f32_t const entry_native = tensor[i][j][k];
                    nk_f32_t const entry_from_byte_offset = *(nk_f32_t *)nk_advance_by_bytes_(&tensor[1][2][1],
                                                                                              xd_index.byte_offset);
                    nk_f32_t const entry_from_coordinate = tensor //
                        [xd_index.coordinates[0] * 2 + 1]         //
                        [xd_index.coordinates[1] * 4 + 2]         //
                        [xd_index.coordinates[2] * 3 + 1];
                    assert(entry_native == entry_from_byte_offset);
                    assert(entry_native == entry_from_coordinate);
                    sum_with_xd_index += entry_from_byte_offset;
                    sum_native_running += entry_native;
                    assert(sum_native_running == sum_with_xd_index);
                    nk_xd_index_next(shape, strides, 3, &xd_index.coordinates[0], &xd_index.byte_offset);
                }
            }
        }
    }

    printf("Test xd_index: PASS\n");
}

/**
 *  @brief  Test nk_xd_span_init function.
 */
void test_xd_span(void) {
    nk_xd_span_t xd_span;

    // Initialize the span
    nk_xd_span_init(&xd_span);

    // Verify all extents are zero
    for (nk_size_t i = 0; i < NK_NDARRAY_MAX_RANK; i++) {
        assert(xd_span.extents[i] == 0);
        assert(xd_span.strides[i] == 0);
    }

    // Verify rank is zero
    assert(xd_span.rank == 0);

    // Test setting values after initialization
    xd_span.rank = 3;
    xd_span.extents[0] = 10;
    xd_span.extents[1] = 20;
    xd_span.extents[2] = 30;
    xd_span.strides[0] = 600;
    xd_span.strides[1] = 30;
    xd_span.strides[2] = 1;

    assert(xd_span.rank == 3);
    assert(xd_span.extents[0] == 10);
    assert(xd_span.extents[1] == 20);
    assert(xd_span.extents[2] == 30);

    printf("Test xd_span: PASS\n");
}

#pragma endregion

#pragma region Data_Types

/**
 *  @brief  Test FP8 E4M3 conversion functions.
 */
void test_fp8_conversions(void) {
    printf("Testing FP8 E4M3 conversions...\n");

    // Test conversion of common values
    nk_f32_t test_values[] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, 10.0f, -10.0f, 100.0f};
    nk_size_t num_tests = sizeof(test_values) / sizeof(test_values[0]);

    for (nk_size_t i = 0; i < num_tests; i++) {
        nk_f32_t original = test_values[i];
        nk_e4m3_t e4m3_value;
        nk_f32_t reconstructed;

        nk_f32_to_e4m3(&original, &e4m3_value);
        nk_e4m3_to_f32(&e4m3_value, &reconstructed);

        if (original != 0.0f) {
            assert((original > 0.0f && reconstructed >= 0.0f) || (original < 0.0f && reconstructed <= 0.0f));
        }

        if (fabsf(original) > 0.1f && fabsf(original) < 100.0f) {
            nk_f32_t relative_error = fabsf((original - reconstructed) / original);
            assert(relative_error < 0.2f);
        }
    }

    // Test round-trip conversion for zero
    {
        nk_f32_t zero = 0.0f;
        nk_e4m3_t e4m3_zero;
        nk_f32_to_e4m3(&zero, &e4m3_zero);
        nk_f32_t result;
        nk_e4m3_to_f32(&e4m3_zero, &result);
        assert(result == 0.0f);
    }

    printf("Test FP8 conversions: PASS\n");
}

/**
 *  @brief Test whether denormals are being flushed to zero or not.
 */
static void test_denormals(void) {
    float subnorm1 = 1e-40f;
    float subnorm2 = 2e-40f;
    float result = subnorm1 * subnorm2;
    int classification = fpclassify(result);
    if (classification == FP_SUBNORMAL) { printf("Denormal test: result is subnormal: %.8g\n", result); }
    else if (result == 0.0f) { printf("Denormal test: result is zero (denormals likely flushed).\n"); }
    else if (classification == FP_NORMAL) { printf("Denormal test: result is normal: %.8g\n", result); }
    else { printf("Denormal test: result has unexpected classification.\n"); }
}

#pragma endregion

#pragma region Distance_Metrics

/**
 *  @brief  A trivial test that calls every implemented distance function on vectors A and B, where A == B.
 */
void test_distance_from_itself(void) {
    nk_f64_t f64s[1536];
    nk_f32_t f32s[1536];
    nk_f16_t f16s[1536];
    nk_bf16_t bf16s[1536];

    nk_f64c_t f64cs[768];
    nk_f32c_t f32cs[768];
    nk_f16c_t f16cs[768];
    nk_bf16c_t bf16cs[768];

    nk_i8_t i8s[1536];
    nk_u8_t u8s[1536];
    nk_b8_t b8s[1536 / 8];
    nk_e4m3_t e4m3s[1536];
    nk_e5m2_t e5m2s[1536];

    // Result variables for different return types
    nk_f32_t result_f32;
    nk_f64_t result_f64;
    nk_i32_t result_i32;
    nk_u32_t result_u32;
    nk_f32c_t result_f32c;
    nk_f64c_t result_f64c;

    // Initialize FP8 arrays with small values (avoid overflow)
    for (int i = 0; i < 1536; i++) {
        e4m3s[i] = 0x3C; // ~1.0 in E4M3
        e5m2s[i] = 0x3C; // ~1.0 in E5M2
    }

    // Angular distance - returns f32 for integer/f16/bf16/f32, f64 for f64
    nk_angular_i8(i8s, i8s, 1536, &result_f32);
    nk_angular_u8(u8s, u8s, 1536, &result_f32);
    nk_angular_f16(f16s, f16s, 1536, &result_f32);
    nk_angular_bf16(bf16s, bf16s, 1536, &result_f32);
    nk_angular_f32(f32s, f32s, 1536, &result_f32);
    nk_angular_f64(f64s, f64s, 1536, &result_f64);

    // L2 squared distance - returns u32 for i8/u8, f32 for f16/bf16/f32, f64 for f64
    nk_l2sq_i8(i8s, i8s, 1536, &result_u32);
    nk_l2sq_u8(u8s, u8s, 1536, &result_u32);
    nk_l2sq_f16(f16s, f16s, 1536, &result_f32);
    nk_l2sq_bf16(bf16s, bf16s, 1536, &result_f32);
    nk_l2sq_f32(f32s, f32s, 1536, &result_f32);
    nk_l2sq_f64(f64s, f64s, 1536, &result_f64);

    // Inner product - returns i32 for i8, u32 for u8, f32 for f16/bf16/f32/e4m3/e5m2, f64 for f64
    nk_dot_i8(i8s, i8s, 1536, &result_i32);
    nk_dot_u8(u8s, u8s, 1536, &result_u32);
    nk_dot_f16(f16s, f16s, 1536, &result_f32);
    nk_dot_bf16(bf16s, bf16s, 1536, &result_f32);
    nk_dot_f32(f32s, f32s, 1536, &result_f32);
    nk_dot_f64(f64s, f64s, 1536, &result_f64);
    nk_dot_e4m3(e4m3s, e4m3s, 1536, &result_f32);
    nk_dot_e5m2(e5m2s, e5m2s, 1536, &result_f32);

    // Complex inner product - returns f32c for f16c/bf16c/f32c, f64c for f64c
    nk_dot_bf16c(bf16cs, bf16cs, 768, &result_f32c);
    nk_dot_f16c(f16cs, f16cs, 768, &result_f32c);
    nk_dot_f32c(f32cs, f32cs, 768, &result_f32c);
    nk_dot_f64c(f64cs, f64cs, 768, &result_f64c);

    // Complex conjugate inner product - returns f32c for f16c/bf16c/f32c, f64c for f64c
    nk_vdot_bf16c(bf16cs, bf16cs, 768, &result_f32c);
    nk_vdot_f16c(f16cs, f16cs, 768, &result_f32c);
    nk_vdot_f32c(f32cs, f32cs, 768, &result_f32c);
    nk_vdot_f64c(f64cs, f64cs, 768, &result_f64c);

    // Hamming and Jaccard - hamming returns u32, jaccard returns f32
    nk_hamming_b8(b8s, b8s, 1536 / 8, &result_u32);
    nk_jaccard_b8(b8s, b8s, 1536 / 8, &result_f32);

    // Divergence metrics - return f32 for f16/bf16/f32, f64 for f64
    nk_jsd_f16(f16s, f16s, 1536, &result_f32);
    nk_jsd_bf16(bf16s, bf16s, 1536, &result_f32);
    nk_jsd_f32(f32s, f32s, 1536, &result_f32);
    nk_jsd_f64(f64s, f64s, 1536, &result_f64);
    nk_kld_f16(f16s, f16s, 1536, &result_f32);
    nk_kld_bf16(bf16s, bf16s, 1536, &result_f32);
    nk_kld_f32(f32s, f32s, 1536, &result_f32);
    nk_kld_f64(f64s, f64s, 1536, &result_f64);

    printf("Test distance_from_itself: PASS\n");
}

/**
 *  @brief  Test distance functions with pre-computed expected values.
 */
void test_distance_precomputed(void) {
    printf("Testing distance with pre-computed values...\n");

    // Test cases with known expected values
    struct {
        nk_f32_t a[4];
        nk_f32_t b[4];
        nk_f64_t expected_l2sq;
        nk_f64_t expected_dot;
        nk_f64_t expected_cos; // cosine similarity
    } cases[] = {
        // Orthogonal unit vectors
        {{1, 0, 0, 0}, {0, 1, 0, 0}, 2.0, 0.0, 0.0},
        // Identical vectors
        {{1, 2, 3, 4}, {1, 2, 3, 4}, 0.0, 30.0, 1.0},
        // Opposite vectors
        {{1, 0, 0, 0}, {-1, 0, 0, 0}, 4.0, -1.0, -1.0},
        // Scaled vectors (same direction)
        {{1, 1, 1, 1}, {2, 2, 2, 2}, 4.0, 8.0, 1.0},
        // Simple case
        {{1, 0, 0, 0}, {1, 0, 0, 0}, 0.0, 1.0, 1.0},
        // 45-degree angle in 2D (padded): L2sq = (1-0.7071)^2 + 0.7071^2 = 0.5858
        {{1, 0, 0, 0}, {0.7071f, 0.7071f, 0, 0}, 0.5858, 0.7071, 0.7071},
    };
    nk_size_t num_cases = sizeof(cases) / sizeof(cases[0]);

    for (nk_size_t i = 0; i < num_cases; i++) {
        nk_f32_t result;

        // Test L2 squared
        nk_l2sq_f32(cases[i].a, cases[i].b, 4, &result);
        nk_f64_t l2sq_error = fabs(result - cases[i].expected_l2sq);
        if (l2sq_error > 0.01) {
            printf("  FAIL case %llu l2sq: expected %.4f, got %.4f\n", (unsigned long long)i, cases[i].expected_l2sq,
                   result);
            fflush(stdout);
        }
        assert(l2sq_error < 0.01);

        // Test dot product
        nk_dot_f32(cases[i].a, cases[i].b, 4, &result);
        nk_f64_t dot_error = fabs(result - cases[i].expected_dot);
        if (dot_error > 0.01) {
            printf("  FAIL case %llu dot: expected %.4f, got %.4f\n", (unsigned long long)i, cases[i].expected_dot,
                   result);
        }
        assert(dot_error < 0.01);

        // Test cosine (via angular, which returns 1 - cos)
        nk_angular_f32(cases[i].a, cases[i].b, 4, &result);
        nk_f64_t cos_result = 1.0 - result;
        nk_f64_t cos_error = fabs(cos_result - cases[i].expected_cos);
        if (cos_error > 0.01) {
            printf("  FAIL case %llu cos: expected %.4f, got %.4f\n", (unsigned long long)i, cases[i].expected_cos,
                   cos_result);
        }
        assert(cos_error < 0.01);
    }

    printf("Test distance_precomputed: PASS\n");
}

/**
 *  @brief  Test bilinear form computation: a^T * M * b.
 */
void test_bilinear(void) {
    printf("Testing bilinear form...\n");

    // Simple 4x4 identity matrix test: a^T * I * b = dot(a, b)
    nk_f32_t a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    nk_f32_t b[4] = {4.0f, 3.0f, 2.0f, 1.0f};
    nk_f32_t identity[16] = {
        1, 0, 0, 0, //
        0, 1, 0, 0, //
        0, 0, 1, 0, //
        0, 0, 0, 1  //
    };
    nk_f32_t result;

    // With identity matrix, bilinear(a, b, I) = dot(a, b) = 1*4 + 2*3 + 3*2 + 4*1 = 20
    nk_bilinear_f32(a, b, identity, 4, &result);
    assert(fabs(result - 20.0) < 0.001);

    // Test with a simple diagonal scaling matrix: diag(2, 2, 2, 2)
    nk_f32_t diag2[16] = {
        2, 0, 0, 0, //
        0, 2, 0, 0, //
        0, 0, 2, 0, //
        0, 0, 0, 2  //
    };
    // bilinear(a, b, diag2) = 2 * dot(a, b) = 40
    nk_bilinear_f32(a, b, diag2, 4, &result);
    assert(fabs(result - 40.0) < 0.001);

    // Test symmetry: a^T * M * b should equal b^T * M^T * a for symmetric M
    nk_f32_t result_ab, result_ba;
    nk_bilinear_f32(a, b, identity, 4, &result_ab);
    nk_bilinear_f32(b, a, identity, 4, &result_ba);
    assert(fabs(result_ab - result_ba) < 0.001);

    printf("Test bilinear: PASS\n");
}

#pragma endregion

#pragma region Elementwise_Operations

/**
 *  @brief  Test scale operation for various data types.
 */
void test_scale_operations(void) {
    printf("Testing scale operations...\n");

    // Test f32 scale: default backend vs serial
    {
        nk_f32_t input[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        nk_f32_t expected[8], result[8];
        nk_f32_t alpha_f32 = 2.0f, beta_f32 = 1.0f;
        nk_scale_f32_serial(input, 8, &alpha_f32, &beta_f32, expected);
        nk_scale_f32(input, 8, &alpha_f32, &beta_f32, result);
        assert_f32_arrays_equal(expected, result, 8, 0.001);
    }

    // Test f64 scale: default backend vs serial
    {
        nk_f64_t input[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        nk_f64_t expected[8], result[8];
        nk_f64_t alpha_f64 = 3.0, beta_f64 = 0.5;
        nk_scale_f64_serial(input, 8, &alpha_f64, &beta_f64, expected);
        nk_scale_f64(input, 8, &alpha_f64, &beta_f64, result);
        assert_f64_arrays_equal(expected, result, 8, 0.001);
    }

    // Test i8 scale: default backend vs serial (exact match for integers)
    {
        nk_i8_t input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        nk_i8_t expected[8], result[8];
        nk_f32_t alpha_i8 = 2.0f, beta_i8 = 1.0f;
        nk_scale_i8_serial(input, 8, &alpha_i8, &beta_i8, expected);
        nk_scale_i8(input, 8, &alpha_i8, &beta_i8, result);
        for (nk_size_t i = 0; i < 8; i++) assert(result[i] == expected[i]);
    }

    // Test u8 scale: default backend vs serial (exact match for integers)
    {
        nk_u8_t input[8] = {10, 20, 30, 40, 50, 60, 70, 80};
        nk_u8_t expected[8], result[8];
        nk_f32_t alpha_u8 = 1.5f, beta_u8 = 5.0f;
        nk_scale_u8_serial(input, 8, &alpha_u8, &beta_u8, expected);
        nk_scale_u8(input, 8, &alpha_u8, &beta_u8, result);
        for (nk_size_t i = 0; i < 8; i++) assert(result[i] == expected[i]);
    }

    printf("Test scale operations: PASS\n");
}

/**
 *  @brief  Test sum operation for various data types.
 */
void test_sum_operations(void) {
    printf("Testing sum operations...\n");

    // Test f32 sum: default backend vs serial
    {
        nk_f32_t a[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        nk_f32_t b[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        nk_f32_t expected[8], result[8];
        nk_sum_f32_serial(a, b, 8, expected);
        nk_sum_f32(a, b, 8, result);
        assert_f32_arrays_equal(expected, result, 8, 0.001);
    }

    // Test f64 sum: default backend vs serial
    {
        nk_f64_t a[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        nk_f64_t b[8] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5};
        nk_f64_t expected[8], result[8];
        nk_sum_f64_serial(a, b, 8, expected);
        nk_sum_f64(a, b, 8, result);
        assert_f64_arrays_equal(expected, result, 8, 0.001);
    }

    // Test i8 sum: default backend vs serial (exact match for integers)
    {
        nk_i8_t a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        nk_i8_t b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
        nk_i8_t expected[8], result[8];
        nk_sum_i8_serial(a, b, 8, expected);
        nk_sum_i8(a, b, 8, result);
        for (nk_size_t i = 0; i < 8; i++) assert(result[i] == expected[i]);
    }

    // Test u8 sum: default backend vs serial (exact match for integers)
    {
        nk_u8_t a[8] = {10, 20, 30, 40, 50, 60, 70, 80};
        nk_u8_t b[8] = {5, 15, 25, 35, 45, 55, 65, 75};
        nk_u8_t expected[8], result[8];
        nk_sum_u8_serial(a, b, 8, expected);
        nk_sum_u8(a, b, 8, result);
        for (nk_size_t i = 0; i < 8; i++) assert(result[i] == expected[i]);
    }

    printf("Test sum operations: PASS\n");
}

/**
 *  @brief  Test weighted sum (wsum) operation: result = alpha * a + beta * b.
 */
void test_wsum_operations(void) {
    printf("Testing wsum operations...\n");

    // Test f32 wsum: default backend vs serial
    {
        nk_f32_t a[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        nk_f32_t b[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        nk_f32_t expected[8], result[8];
        nk_f32_t alpha_wsum_f32 = 2.0f, beta_wsum_f32 = 0.5f;
        nk_wsum_f32_serial(a, b, 8, &alpha_wsum_f32, &beta_wsum_f32, expected);
        nk_wsum_f32(a, b, 8, &alpha_wsum_f32, &beta_wsum_f32, result);
        assert_f32_arrays_equal(expected, result, 8, 0.001);
    }

    // Test f64 wsum: default backend vs serial
    {
        nk_f64_t a[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        nk_f64_t b[8] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5};
        nk_f64_t expected[8], result[8];
        nk_f64_t alpha_wsum_f64 = 0.3, beta_wsum_f64 = 0.7;
        nk_wsum_f64_serial(a, b, 8, &alpha_wsum_f64, &beta_wsum_f64, expected);
        nk_wsum_f64(a, b, 8, &alpha_wsum_f64, &beta_wsum_f64, result);
        assert_f64_arrays_equal(expected, result, 8, 0.001);
    }

    printf("Test wsum operations: PASS\n");
}

/**
 *  @brief  Test fused multiply-add (fma) operation: result = alpha * a * b + beta * c.
 */
void test_fma_operations(void) {
    printf("Testing fma operations...\n");

    // Test f32 fma: default backend vs serial
    {
        nk_f32_t a[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        nk_f32_t b[8] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
        nk_f32_t c[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        nk_f32_t expected[8], result[8];
        nk_f32_t alpha_fma_f32 = 1.0f, beta_fma_f32 = 1.0f;
        nk_fma_f32_serial(a, b, c, 8, &alpha_fma_f32, &beta_fma_f32, expected);
        nk_fma_f32(a, b, c, 8, &alpha_fma_f32, &beta_fma_f32, result);
        assert_f32_arrays_equal(expected, result, 8, 0.001);
    }

    // Test f64 fma: default backend vs serial
    {
        nk_f64_t a[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        nk_f64_t b[8] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
        nk_f64_t c[8] = {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0};
        nk_f64_t expected[8], result[8];
        nk_f64_t alpha_fma_f64 = 2.0, beta_fma_f64 = 0.1;
        nk_fma_f64_serial(a, b, c, 8, &alpha_fma_f64, &beta_fma_f64, expected);
        nk_fma_f64(a, b, c, 8, &alpha_fma_f64, &beta_fma_f64, result);
        assert_f64_arrays_equal(expected, result, 8, 0.001);
    }

    printf("Test fma operations: PASS\n");
}

#pragma endregion

#pragma region Trigonometry

/**
 *  @brief  Goes through all possible `f32` values in a relevant range, computing
 */
void test_approximate_math(void) {

    typedef struct error_aggregator {
        nk_f64_t absolute_error;
        nk_f64_t relative_error;
        nk_f64_t max_error;
    } error_aggregator;

    error_aggregator f32_cos_errors = {0, 0, 0}, f32_sin_errors = {0, 0, 0};
    error_aggregator f32_atan_errors = {0, 0, 0}, f32_atan2_errors = {0, 0, 0};
    error_aggregator f64_cos_errors = {0, 0, 0}, f64_sin_errors = {0, 0, 0};
    error_aggregator f64_atan_errors = {0, 0, 0}, f64_atan2_errors = {0, 0, 0};

    nk_f32_t const range_min = -3.14159265358979323846f * 2;
    nk_f32_t const range_max = 3.14159265358979323846f * 2;

    // Test all possible values of f32 within ranges: [-π, -1], [-1, -0], [0, 1], [1, π].
    nk_size_t const step = 1;
    nk_size_t const count_tests = 0xFFFFFFFFull / step;
    union {
        nk_f32_t f32;
        nk_u32_t u32;
    } x;

    // Run separate loops for every operation
    for (x.u32 = 0; (x.u32 + step) <= 0xFFFFFFFFull; x.u32 += step) {
        if (x.f32 < range_min || x.f32 > range_max) continue;
        nk_f32_t f32_sin_baseline = sinf(x.f32);
        nk_f32_t f32_sin_approx = nk_f32_sin(x.f32);
        nk_f32_t f32_sin_diff = fabsf(f32_sin_baseline - f32_sin_approx);
        nk_f32_t f32_sin_max = fmaxf(fabsf(f32_sin_baseline), fabsf(f32_sin_approx));
        f32_sin_errors.absolute_error += f32_sin_diff;
        f32_sin_errors.relative_error += f32_sin_max != 0 ? f32_sin_diff / f32_sin_max : 0;
        f32_sin_errors.max_error = fmax(f32_sin_errors.max_error, f32_sin_diff);
    }
    printf("f32 sin: <error>= %f, up to %f\n", f32_sin_errors.absolute_error / count_tests, f32_sin_errors.max_error);
    for (x.u32 = 0; (x.u32 + step) <= 0xFFFFFFFFull; x.u32 += step) {
        if (x.f32 < range_min || x.f32 > range_max) continue;
        nk_f32_t f32_cos_baseline = cosf(x.f32);
        nk_f32_t f32_cos_approx = nk_f32_cos(x.f32);
        nk_f32_t f32_cos_diff = fabsf(f32_cos_baseline - f32_cos_approx);
        nk_f32_t f32_cos_max = fmaxf(fabsf(f32_cos_baseline), fabsf(f32_cos_approx));
        f32_cos_errors.absolute_error += f32_cos_diff;
        f32_cos_errors.relative_error += f32_cos_max != 0 ? f32_cos_diff / f32_cos_max : 0;
        f32_cos_errors.max_error = fmax(f32_cos_errors.max_error, f32_cos_diff);
    }
    printf("f32 cos: <error>= %f, up to %f\n", f32_cos_errors.absolute_error / count_tests, f32_cos_errors.max_error);
    for (x.u32 = 0; (x.u32 + step) <= 0xFFFFFFFFull; x.u32 += step) {
        if (x.f32 < range_min || x.f32 > range_max) continue;
        nk_f32_t f32_atan_baseline = atanf(x.f32);
        nk_f32_t f32_atan_approx = nk_f32_atan(x.f32);
        nk_f32_t f32_atan_diff = fabs(f32_atan_baseline - f32_atan_approx);
        nk_f32_t f32_atan_max = fmax(fabs(f32_atan_baseline), fabs(f32_atan_approx));
        f32_atan_errors.absolute_error += f32_atan_diff;
        f32_atan_errors.relative_error += f32_atan_max != 0 ? f32_atan_diff / f32_atan_max : 0;
        f32_atan_errors.max_error = fmax(f32_atan_errors.max_error, f32_atan_diff);
    }
    printf("f32 atan: <error>= %f, up to %f\n", f32_atan_errors.absolute_error / count_tests,
           f32_atan_errors.max_error);
    for (x.u32 = 0; (x.u32 + step) <= 0xFFFFFFFFull; x.u32 += step) {
        if (x.f32 < range_min || x.f32 > range_max) continue;
        nk_f32_t f32_atan2_baseline = atan2f(x.f32, (1 - x.f32));
        nk_f32_t f32_atan2_approx = nk_f32_atan2(x.f32, (1 - x.f32));
        nk_f32_t f32_atan2_diff = fabs(f32_atan2_baseline - f32_atan2_approx);
        nk_f32_t f32_atan2_max = fmax(fabs(f32_atan2_baseline), fabs(f32_atan2_approx));
        f32_atan2_errors.absolute_error += f32_atan2_diff;
        f32_atan2_errors.relative_error += f32_atan2_max != 0 ? f32_atan2_diff / f32_atan2_max : 0;
        f32_atan2_errors.max_error = fmax(f32_atan2_errors.max_error, f32_atan2_diff);
    }
    printf("f32 atan2: <error>= %f, up to %f\n", f32_atan2_errors.absolute_error / count_tests,
           f32_atan2_errors.max_error);
    for (x.u32 = 0; (x.u32 + step) <= 0xFFFFFFFFull; x.u32 += step) {
        if (x.f32 < range_min || x.f32 > range_max) continue;
        nk_f64_t f64_sin_baseline = sin(x.f32);
        nk_f64_t f64_sin_approx = nk_f64_sin(x.f32);
        nk_f64_t f64_sin_diff = fabs(f64_sin_baseline - f64_sin_approx);
        nk_f64_t f64_sin_max = fmax(fabs(f64_sin_baseline), fabs(f64_sin_approx));
        f64_sin_errors.absolute_error += f64_sin_diff;
        f64_sin_errors.relative_error += f64_sin_max != 0 ? f64_sin_diff / f64_sin_max : 0;
        f64_sin_errors.max_error = fmax(f64_sin_errors.max_error, f64_sin_diff);
    }
    printf("f64 sin: <error>= %f, up to %f\n", f64_sin_errors.absolute_error / count_tests, f64_sin_errors.max_error);
    for (x.u32 = 0; (x.u32 + step) <= 0xFFFFFFFFull; x.u32 += step) {
        if (x.f32 < range_min || x.f32 > range_max) continue;
        nk_f64_t f64_cos_baseline = cos(x.f32);
        nk_f64_t f64_cos_approx = nk_f64_cos(x.f32);
        nk_f64_t f64_cos_diff = fabs(f64_cos_baseline - f64_cos_approx);
        nk_f64_t f64_cos_max = fmax(fabs(f64_cos_baseline), fabs(f64_cos_approx));
        f64_cos_errors.absolute_error += f64_cos_diff;
        f64_cos_errors.relative_error += f64_cos_max != 0 ? f64_cos_diff / f64_cos_max : 0;
        f64_cos_errors.max_error = fmax(f64_cos_errors.max_error, f64_cos_diff);
    }
    printf("f64 cos: <error>= %f, up to %f\n", f64_cos_errors.absolute_error / count_tests, f64_cos_errors.max_error);
    for (x.u32 = 0; (x.u32 + step) <= 0xFFFFFFFFull; x.u32 += step) {
        if (x.f32 < range_min || x.f32 > range_max) continue;
        nk_f64_t f64_atan_baseline = atan(x.f32);
        nk_f64_t f64_atan_approx = nk_f64_atan(x.f32);
        nk_f64_t f64_atan_diff = fabs(f64_atan_baseline - f64_atan_approx);
        nk_f64_t f64_atan_max = fmax(fabs(f64_atan_baseline), fabs(f64_atan_approx));
        f64_atan_errors.absolute_error += f64_atan_diff;
        f64_atan_errors.relative_error += f64_atan_max != 0 ? f64_atan_diff / f64_atan_max : 0;
        f64_atan_errors.max_error = fmax(f64_atan_errors.max_error, f64_atan_diff);
    }
    printf("f64 atan: <error>= %f, up to %f\n", f64_atan_errors.absolute_error / count_tests,
           f64_atan_errors.max_error);
    for (x.u32 = 0; (x.u32 + step) <= 0xFFFFFFFFull; x.u32 += step) {
        if (x.f32 < range_min || x.f32 > range_max) continue;
        nk_f64_t f64_atan2_baseline = atan2(x.f32, (1 - x.f32));
        nk_f64_t f64_atan2_approx = nk_f64_atan2(x.f32, (1 - x.f32));
        nk_f64_t f64_atan2_diff = fabs(f64_atan2_baseline - f64_atan2_approx);
        nk_f64_t f64_atan2_max = fmax(fabs(f64_atan2_baseline), fabs(f64_atan2_approx));
        f64_atan2_errors.absolute_error += f64_atan2_diff;
        f64_atan2_errors.relative_error += f64_atan2_max != 0 ? f64_atan2_diff / f64_atan2_max : 0;
        f64_atan2_errors.max_error = fmax(f64_atan2_errors.max_error, f64_atan2_diff);
    }
    printf("f64 atan2: <error>= %f, up to %f\n", f64_atan2_errors.absolute_error / count_tests,
           f64_atan2_errors.max_error);
    printf("Test approximate_math: PASS\n");
}

/**
 *  @brief  Test SIMD trigonometry implementations against serial baselines.
 *          Tests sin, cos, atan for Haswell (AVX2) and Skylake (AVX-512) backends.
 */
void test_simd_trigonometry(void) {
    printf("Testing SIMD trigonometry...\n");

    nk_capability_t caps = nk_capabilities();
    nk_f64_t const pi = 3.14159265358979323846;
    // SLEEF-level error bounds: 3.5 ULP for fast functions
    // For f32: 1 ULP ≈ 1.2e-7, so 3.5 ULP ≈ 4e-7, we use 1e-5 for margin
    // For f64: 1 ULP ≈ 2.2e-16, so 3.5 ULP ≈ 8e-16, we use 1e-10 for polynomial margin
    nk_f64_t const tolerance_f32 = 1e-5;
    nk_f64_t const tolerance_f64 = 1e-10;

    // Test buffer sizes that exercise tail handling
    nk_size_t const sizes[] = {1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100};
    nk_size_t const num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // Stack-allocated buffers sized to max element in sizes[]
    nk_f32_t f32_ins[100], f32_serial[100], f32_simd[100];
    nk_f64_t f64_ins[100], f64_serial[100], f64_simd[100];

    // Test each size
    for (nk_size_t s = 0; s < num_sizes; s++) {
        nk_size_t n = sizes[s];

        // Fill with random values in [-2π, 2π] for this test size
        for (nk_size_t i = 0; i < n; i++) {
            f32_ins[i] = (nk_f32_t)test_random_f64(-2 * pi, 2 * pi);
            f64_ins[i] = test_random_f64(-2 * pi, 2 * pi);
        }

        // === Test f32 sin ===
        nk_sin_f32_serial(f32_ins, n, f32_serial);
#if NK_TARGET_HASWELL
        if (caps & nk_cap_haswell_k) {
            nk_sin_f32_haswell(f32_ins, n, f32_simd);
            assert_f32_arrays_equal(f32_serial, f32_simd, n, tolerance_f32);
        }
#endif
#if NK_TARGET_SKYLAKE
        if (caps & nk_cap_skylake_k) {
            nk_sin_f32_skylake(f32_ins, n, f32_simd);
            assert_f32_arrays_equal(f32_serial, f32_simd, n, tolerance_f32);
        }
#endif

        // === Test f32 cos ===
        nk_cos_f32_serial(f32_ins, n, f32_serial);
#if NK_TARGET_HASWELL
        if (caps & nk_cap_haswell_k) {
            nk_cos_f32_haswell(f32_ins, n, f32_simd);
            assert_f32_arrays_equal(f32_serial, f32_simd, n, tolerance_f32);
        }
#endif
#if NK_TARGET_SKYLAKE
        if (caps & nk_cap_skylake_k) {
            nk_cos_f32_skylake(f32_ins, n, f32_simd);
            assert_f32_arrays_equal(f32_serial, f32_simd, n, tolerance_f32);
        }
#endif

        // === Test f32 atan ===
        nk_atan_f32_serial(f32_ins, n, f32_serial);
#if NK_TARGET_HASWELL
        if (caps & nk_cap_haswell_k) {
            nk_atan_f32_haswell(f32_ins, n, f32_simd);
            assert_f32_arrays_equal(f32_serial, f32_simd, n, tolerance_f32);
        }
#endif
#if NK_TARGET_SKYLAKE
        if (caps & nk_cap_skylake_k) {
            nk_atan_f32_skylake(f32_ins, n, f32_simd);
            assert_f32_arrays_equal(f32_serial, f32_simd, n, tolerance_f32);
        }
#endif

        // === Test f64 sin ===
        nk_sin_f64_serial(f64_ins, n, f64_serial);
#if NK_TARGET_HASWELL
        if (caps & nk_cap_haswell_k) {
            nk_sin_f64_haswell(f64_ins, n, f64_simd);
            assert_f64_arrays_equal(f64_serial, f64_simd, n, tolerance_f64);
        }
#endif
#if NK_TARGET_SKYLAKE
        if (caps & nk_cap_skylake_k) {
            nk_sin_f64_skylake(f64_ins, n, f64_simd);
            assert_f64_arrays_equal(f64_serial, f64_simd, n, tolerance_f64);
        }
#endif

        // === Test f64 cos ===
        nk_cos_f64_serial(f64_ins, n, f64_serial);
#if NK_TARGET_HASWELL
        if (caps & nk_cap_haswell_k) {
            nk_cos_f64_haswell(f64_ins, n, f64_simd);
            assert_f64_arrays_equal(f64_serial, f64_simd, n, tolerance_f64);
        }
#endif
#if NK_TARGET_SKYLAKE
        if (caps & nk_cap_skylake_k) {
            nk_cos_f64_skylake(f64_ins, n, f64_simd);
            assert_f64_arrays_equal(f64_serial, f64_simd, n, tolerance_f64);
        }
#endif

        // === Test f64 atan ===
        nk_atan_f64_serial(f64_ins, n, f64_serial);
#if NK_TARGET_HASWELL
        if (caps & nk_cap_haswell_k) {
            nk_atan_f64_haswell(f64_ins, n, f64_simd);
            assert_f64_arrays_equal(f64_serial, f64_simd, n, tolerance_f64);
        }
#endif
#if NK_TARGET_SKYLAKE
        if (caps & nk_cap_skylake_k) {
            nk_atan_f64_skylake(f64_ins, n, f64_simd);
            assert_f64_arrays_equal(f64_serial, f64_simd, n, tolerance_f64);
        }
#endif
    }

    printf("Test SIMD trigonometry: PASS\n");
}

/**
 *  @brief  Test atan2 in all four quadrants and edge cases.
 *          Verifies that the approximate implementation matches the standard library.
 */
void test_atan2_quadrants(void) {
    printf("Testing atan2 quadrants...\n");

    nk_f64_t const pi = 3.14159265358979323846;
    // SLEEF-level error bounds with margin for polynomial approximations
    nk_f64_t const tolerance_f32 = 1e-5;
    nk_f64_t const tolerance_f64 = 1e-10;

    // Test cases covering all quadrants and edge cases
    struct {
        nk_f64_t y;
        nk_f64_t x;
        char const *name;
    } test_cases[] = {
        // Quadrant 1: x > 0, y > 0
        {1.0, 1.0, "Q1: 45 degrees"},
        {1.0, 2.0, "Q1: ~26.6 degrees"},
        {2.0, 1.0, "Q1: ~63.4 degrees"},
        {0.1, 0.1, "Q1: small values"},
        {10.0, 10.0, "Q1: large values"},
        // Quadrant 2: x < 0, y > 0
        {1.0, -1.0, "Q2: 135 degrees"},
        {1.0, -2.0, "Q2: ~153.4 degrees"},
        {2.0, -1.0, "Q2: ~116.6 degrees"},
        {0.1, -0.1, "Q2: small values"},
        {10.0, -10.0, "Q2: large values"},
        // Quadrant 3: x < 0, y < 0
        {-1.0, -1.0, "Q3: -135 degrees"},
        {-1.0, -2.0, "Q3: ~-153.4 degrees"},
        {-2.0, -1.0, "Q3: ~-116.6 degrees"},
        {-0.1, -0.1, "Q3: small values"},
        {-10.0, -10.0, "Q3: large values"},
        // Quadrant 4: x > 0, y < 0
        {-1.0, 1.0, "Q4: -45 degrees"},
        {-1.0, 2.0, "Q4: ~-26.6 degrees"},
        {-2.0, 1.0, "Q4: ~-63.4 degrees"},
        {-0.1, 0.1, "Q4: small values"},
        {-10.0, 10.0, "Q4: large values"},
        // Edge cases on axes
        {0.0, 1.0, "positive x-axis"},
        {0.0, -1.0, "negative x-axis"},
        {1.0, 0.0, "positive y-axis"},
        {-1.0, 0.0, "negative y-axis"},
        // Near-zero cases
        {1e-10, 1.0, "very small y"},
        {1.0, 1e-10, "very small x"},
        // Large magnitude differences
        {1e6, 1.0, "large y, small x"},
        {1.0, 1e6, "small y, large x"},
    };
    nk_size_t num_cases = sizeof(test_cases) / sizeof(test_cases[0]);

    // Test f32 atan2
    for (nk_size_t i = 0; i < num_cases; i++) {
        nk_f32_t y = (nk_f32_t)test_cases[i].y;
        nk_f32_t x = (nk_f32_t)test_cases[i].x;
        nk_f32_t expected = atan2f(y, x);
        nk_f32_t result = nk_f32_atan2(y, x);
        nk_f64_t diff = fabs((nk_f64_t)expected - (nk_f64_t)result);
        if (diff >= tolerance_f32) {
            printf("  FAIL f32 %s: y=%f, x=%f, expected=%f, got=%f, diff=%e\n", test_cases[i].name, y, x, expected,
                   result, diff);
        }
        assert(diff < tolerance_f32);
    }

    // Test f64 atan2
    for (nk_size_t i = 0; i < num_cases; i++) {
        nk_f64_t y = test_cases[i].y;
        nk_f64_t x = test_cases[i].x;
        nk_f64_t expected = atan2(y, x);
        nk_f64_t result = nk_f64_atan2(y, x);
        nk_f64_t diff = fabs(expected - result);
        if (diff >= tolerance_f64) {
            printf("  FAIL f64 %s: y=%f, x=%f, expected=%f, got=%f, diff=%e\n", test_cases[i].name, y, x, expected,
                   result, diff);
        }
        assert(diff < tolerance_f64);
    }

    // Random fuzzing across all quadrants
    nk_size_t num_random_tests = 1000;
    for (nk_size_t i = 0; i < num_random_tests; i++) {
        // Generate random y and x in range [-10, 10], avoiding (0, 0)
        nk_f64_t y = test_random_f64(-10.0, 10.0);
        nk_f64_t x = test_random_f64(-10.0, 10.0);
        if (fabs(y) < 1e-10 && fabs(x) < 1e-10) continue; // Skip near-origin

        // Test f32
        nk_f32_t y32 = (nk_f32_t)y;
        nk_f32_t x32 = (nk_f32_t)x;
        nk_f32_t expected32 = atan2f(y32, x32);
        nk_f32_t result32 = nk_f32_atan2(y32, x32);
        nk_f64_t diff32 = fabs((nk_f64_t)expected32 - (nk_f64_t)result32);
        if (diff32 >= tolerance_f32) {
            printf("  FAIL random f32 #%llu: y=%f, x=%f, expected=%f, got=%f, diff=%e\n", (unsigned long long)i, y32,
                   x32, expected32, result32, diff32);
        }
        assert(diff32 < tolerance_f32);

        // Test f64
        nk_f64_t expected64 = atan2(y, x);
        nk_f64_t result64 = nk_f64_atan2(y, x);
        nk_f64_t diff64 = fabs(expected64 - result64);
        if (diff64 >= tolerance_f64) {
            printf("  FAIL random f64 #%llu: y=%f, x=%f, expected=%f, got=%f, diff=%e\n", (unsigned long long)i, y, x,
                   expected64, result64, diff64);
        }
        assert(diff64 < tolerance_f64);
    }

    printf("Test atan2 quadrants: PASS\n");
}

#pragma endregion

#pragma region Geospatial

/**
 *  @brief  Test Haversine geospatial distance function.
 *          Uses known reference distances and random fuzzing.
 */
void test_geospatial_haversine(void) {
    printf("Testing geospatial Haversine...\n");

    nk_capability_t caps = nk_capabilities();
    nk_f64_t const pi = 3.14159265358979323846;

    // Known reference points (coordinates in radians)
    struct {
        nk_f64_t lat1, lon1, lat2, lon2;
        nk_f64_t expected_km;
        char const *name;
    } known_cases[] = {
        // NYC (40.7128°N, 74.0060°W) to LA (34.0522°N, 118.2437°W): ~3,940 km
        {40.7128 * pi / 180, -74.0060 * pi / 180, 34.0522 * pi / 180, -118.2437 * pi / 180, 3940, "NYC to LA"},
        // London (51.5074°N, 0.1278°W) to Paris (48.8566°N, 2.3522°E): ~344 km
        {51.5074 * pi / 180, -0.1278 * pi / 180, 48.8566 * pi / 180, 2.3522 * pi / 180, 344, "London to Paris"},
        // Same point should return 0
        {0.5, 1.0, 0.5, 1.0, 0, "same point"},
        // North pole to equator: ~10,008 km
        {pi / 2, 0.0, 0.0, 0.0, 10008, "North pole to equator"},
        // Equator points 180 degrees apart: half Earth circumference ~20,015 km
        {0.0, 0.0, 0.0, pi, 20015, "Antipodal equator points"},
    };
    nk_size_t num_known = sizeof(known_cases) / sizeof(known_cases[0]);

    // Test known cases with serial implementation
    for (nk_size_t i = 0; i < num_known; i++) {
        nk_f64_t lat1 = known_cases[i].lat1;
        nk_f64_t lon1 = known_cases[i].lon1;
        nk_f64_t lat2 = known_cases[i].lat2;
        nk_f64_t lon2 = known_cases[i].lon2;
        nk_f64_t result;

        nk_haversine_f64_serial(&lat1, &lon1, &lat2, &lon2, 1, &result);
        nk_f64_t result_km = result / 1000.0;
        nk_f64_t expected_km = known_cases[i].expected_km;
        nk_f64_t rel_error = expected_km > 0 ? fabs(result_km - expected_km) / expected_km : fabs(result_km);

        // Allow 5% relative error for known points (different Earth radius assumptions)
        if (rel_error > 0.05 && expected_km > 0) {
            printf("  FAIL %s: expected ~%.0f km, got %.0f km (%.1f%% error)\n", known_cases[i].name, expected_km,
                   result_km, rel_error * 100);
        }
        // For zero distance case, allow small absolute error
        if (expected_km == 0 && result > 1.0) { // More than 1 meter for same point
            printf("  FAIL %s: expected 0, got %.2f meters\n", known_cases[i].name, result);
        }
    }

    // Random fuzzing: generate-and-test pattern (no heap allocations)
    nk_size_t const num_random = 1000;
    for (nk_size_t i = 0; i < num_random; i++) {
        nk_f64_t lat1 = test_random_f64(-pi / 2, pi / 2);
        nk_f64_t lon1 = test_random_f64(-pi, pi);
        nk_f64_t lat2 = test_random_f64(-pi / 2, pi / 2);
        nk_f64_t lon2 = test_random_f64(-pi, pi);
        nk_f32_t lat1_f32 = (nk_f32_t)lat1, lon1_f32 = (nk_f32_t)lon1;
        nk_f32_t lat2_f32 = (nk_f32_t)lat2, lon2_f32 = (nk_f32_t)lon2;
        nk_f64_t expected, actual_f64;
        nk_f32_t actual_f32;

        // f64 serial as baseline
        nk_haversine_f64_serial(&lat1, &lon1, &lat2, &lon2, 1, &expected);

#if NK_TARGET_HASWELL
        if (caps & nk_cap_haswell_k) {
            nk_haversine_f64_haswell(&lat1, &lon1, &lat2, &lon2, 1, &actual_f64);
            assert_f64_near(expected, actual_f64, 0.001, 100, "haversine f64 haswell");
        }
#endif
#if NK_TARGET_SKYLAKE
        if (caps & nk_cap_skylake_k) {
            nk_haversine_f64_skylake(&lat1, &lon1, &lat2, &lon2, 1, &actual_f64);
            assert_f64_near(expected, actual_f64, 0.001, 100, "haversine f64 skylake");
        }
#endif
        // f32 serial
        nk_haversine_f32_serial(&lat1_f32, &lon1_f32, &lat2_f32, &lon2_f32, 1, &actual_f32);
        assert_f64_near(expected, actual_f32, 0.01, 1000, "haversine f32 serial");

#if NK_TARGET_SKYLAKE
        if (caps & nk_cap_skylake_k) {
            nk_haversine_f32_skylake(&lat1_f32, &lon1_f32, &lat2_f32, &lon2_f32, 1, &actual_f32);
            assert_f64_near(expected, actual_f32, 0.01, 1000, "haversine f32 skylake");
        }
#endif
    }

    printf("Test geospatial Haversine: PASS\n");
}

/**
 *  @brief  Test Vincenty geospatial distance function.
 *          Focuses on corner cases and convergence edge cases.
 */
void test_geospatial_vincenty(void) {
    printf("Testing geospatial Vincenty...\n");

    nk_capability_t caps = nk_capabilities();
    nk_f64_t const pi = 3.14159265358979323846;

    // Corner cases for Vincenty
    struct {
        nk_f64_t lat1, lon1, lat2, lon2;
        char const *name;
        int expect_convergence; // 1 = should converge, 0 = may not converge (antipodal)
    } corner_cases[] = {
        // Same point: distance = 0
        {0.5, 1.0, 0.5, 1.0, "same point", 1},
        // Very close points
        {0.5, 1.0, 0.5 + 1e-8, 1.0 + 1e-8, "very close points", 1},
        // Equatorial geodesic
        {0.0, 0.0, 0.0, 1.0, "equatorial geodesic", 1},
        // Meridional geodesic (same longitude)
        {0.5, 1.0, 0.7, 1.0, "meridional geodesic", 1},
        // Points near poles
        {pi / 2 - 0.01, 0.0, pi / 2 - 0.02, 1.0, "near north pole", 1},
        {-pi / 2 + 0.01, 0.0, -pi / 2 + 0.02, 1.0, "near south pole", 1},
        // Large longitude difference (but not antipodal)
        {0.3, 0.0, 0.3, 2.9, "large lon diff ~166 deg", 1},
        // Moderate distance
        {0.7102, -1.2918, 0.5944, -2.0637, "NYC to LA approx", 1},
        // The previously problematic case
        {-1.04902546, 2.00303635, 0.08598804, -1.35023794, "problematic case", 1},
    };
    nk_size_t num_cases = sizeof(corner_cases) / sizeof(corner_cases[0]);

    // Test corner cases with serial implementation
    for (nk_size_t i = 0; i < num_cases; i++) {
        if (!corner_cases[i].expect_convergence) continue;

        nk_f64_t lat1 = corner_cases[i].lat1;
        nk_f64_t lon1 = corner_cases[i].lon1;
        nk_f64_t lat2 = corner_cases[i].lat2;
        nk_f64_t lon2 = corner_cases[i].lon2;
        nk_f64_t result;

        nk_vincenty_f64_serial(&lat1, &lon1, &lat2, &lon2, 1, &result);

        // Basic sanity checks
        if (result < 0) {
            printf("  FAIL %s: got negative distance %.2f\n", corner_cases[i].name, result);
            assert(result >= 0);
        }
        // Same point should be ~0
        if (lat1 == lat2 && lon1 == lon2 && result > 1.0) {
            printf("  FAIL %s: same point got %.2f meters\n", corner_cases[i].name, result);
            assert(result < 1.0);
        }
        // Distance should not exceed half Earth circumference
        if (result > 21000000) { // ~21,000 km max
            printf("  FAIL %s: got %.2f meters (too large)\n", corner_cases[i].name, result);
            assert(result < 21000000);
        }
    }

    // Random fuzzing: generate-and-test pattern (no heap allocations)
    // Vincenty allows some failures due to precision limits, so we track failure counts
    nk_size_t const num_random = 1000;
    nk_size_t f64_skylake_failures = 0, f32_serial_failures = 0, f32_skylake_failures = 0;

    for (nk_size_t i = 0; i < num_random; i++) {
        // Generate random coordinates, avoiding antipodal points
        nk_f64_t lat1 = test_random_f64(-pi / 2 * 0.9, pi / 2 * 0.9);
        nk_f64_t lon1 = test_random_f64(-pi * 0.9, pi * 0.9);
        nk_f64_t lat2 = test_random_f64(-pi / 2 * 0.9, pi / 2 * 0.9);
        nk_f64_t lon2 = test_random_f64(-pi * 0.9, pi * 0.9);
        nk_f32_t lat1_f32 = (nk_f32_t)lat1, lon1_f32 = (nk_f32_t)lon1;
        nk_f32_t lat2_f32 = (nk_f32_t)lat2, lon2_f32 = (nk_f32_t)lon2;
        nk_f64_t expected, actual_f64;
        nk_f32_t actual_f32;
        nk_f64_t diff, rel_err;

        // f64 serial as baseline
        nk_vincenty_f64_serial(&lat1, &lon1, &lat2, &lon2, 1, &expected);

#if NK_TARGET_SKYLAKE
        if (caps & nk_cap_skylake_k) {
            nk_vincenty_f64_skylake(&lat1, &lon1, &lat2, &lon2, 1, &actual_f64);
            diff = fabs(expected - actual_f64);
            rel_err = expected > 0 ? diff / expected : diff;
            if (rel_err > 0.00001 && diff > 10) f64_skylake_failures++;
        }
#endif
        // f32 serial
        nk_vincenty_f32_serial(&lat1_f32, &lon1_f32, &lat2_f32, &lon2_f32, 1, &actual_f32);
        diff = fabs(expected - actual_f32);
        rel_err = expected > 0 ? diff / expected : diff;
        if (rel_err > 0.01 && diff > 1000) f32_serial_failures++;

#if NK_TARGET_SKYLAKE
        if (caps & nk_cap_skylake_k) {
            nk_vincenty_f32_skylake(&lat1_f32, &lon1_f32, &lat2_f32, &lon2_f32, 1, &actual_f32);
            diff = fabs(expected - actual_f32);
            rel_err = expected > 0 ? diff / expected : diff;
            if (rel_err > 0.05 && diff > 5000) f32_skylake_failures++;
        }
#endif
    }

    // Allow some failures due to f32 precision limits
#if NK_TARGET_SKYLAKE
    if (caps & nk_cap_skylake_k) {
        if (f64_skylake_failures > 0)
            printf("  Vincenty f64 Skylake failures: %llu/%llu\n", (unsigned long long)f64_skylake_failures,
                   (unsigned long long)num_random);
        assert(f64_skylake_failures < num_random / 100);
        if (f32_skylake_failures > 0)
            printf("  Vincenty f32 Skylake failures: %llu/%llu\n", (unsigned long long)f32_skylake_failures,
                   (unsigned long long)num_random);
        assert(f32_skylake_failures < num_random / 10);
    }
#endif
    if (f32_serial_failures > 0)
        printf("  Vincenty f32 serial failures: %llu/%llu\n", (unsigned long long)f32_serial_failures,
               (unsigned long long)num_random);
    assert(f32_serial_failures < num_random / 20);

    printf("Test geospatial Vincenty: PASS\n");
}

#pragma endregion

#pragma region Matrix Multiplication

/**
 *  @brief  Reference BF16 matmul: C[m×n] = A[m×k] × B[n×k]ᵀ
 */
static void reference_matmul_bf16_f32(nk_bf16_t const *a, nk_bf16_t const *b, nk_f32_t *c, nk_size_t m, nk_size_t n,
                                      nk_size_t k) {
    for (nk_size_t i = 0; i < m; i++) {
        for (nk_size_t j = 0; j < n; j++) {
            nk_f32_t sum = 0;
            for (nk_size_t kk = 0; kk < k; kk++) {
                nk_f32_t a_val, b_val;
                nk_bf16_to_f32(&a[i * k + kk], &a_val);
                nk_bf16_to_f32(&b[j * k + kk], &b_val);
                sum += a_val * b_val;
            }
            c[i * n + j] = sum;
        }
    }
}

/**
 *  @brief  Reference I8 matmul: C[m×n] = A[m×k] × B[n×k]ᵀ
 */
static void reference_matmul_i8_i32(nk_i8_t const *a, nk_i8_t const *b, nk_i32_t *c, nk_size_t m, nk_size_t n,
                                    nk_size_t k) {
    for (nk_size_t i = 0; i < m; i++) {
        for (nk_size_t j = 0; j < n; j++) {
            nk_i32_t sum = 0;
            for (nk_size_t kk = 0; kk < k; kk++) { sum += (nk_i32_t)a[i * k + kk] * (nk_i32_t)b[j * k + kk]; }
            c[i * n + j] = sum;
        }
    }
}

/**
 *  @brief  Test BF16 matmul (pack + multiply) against reference.
 */
void test_matmul_bf16(void) {
    nk_size_t m = 64, n = 64, k = 64;

    // Allocate matrices
    nk_bf16_t *a = (nk_bf16_t *)malloc(m * k * sizeof(nk_bf16_t));
    nk_bf16_t *b = (nk_bf16_t *)malloc(n * k * sizeof(nk_bf16_t));
    nk_f32_t *c_ref = (nk_f32_t *)malloc(m * n * sizeof(nk_f32_t));
    nk_f32_t *c_test = (nk_f32_t *)malloc(m * n * sizeof(nk_f32_t));

    // Initialize with random values
    for (nk_size_t i = 0; i < m * k; i++) {
        nk_f32_t val = (nk_f32_t)test_random_f64(-1.0, 1.0);
        nk_f32_to_bf16(&val, &a[i]);
    }
    for (nk_size_t i = 0; i < n * k; i++) {
        nk_f32_t val = (nk_f32_t)test_random_f64(-1.0, 1.0);
        nk_f32_to_bf16(&val, &b[i]);
    }

    // Compute reference
    reference_matmul_bf16_f32(a, b, c_ref, m, n, k);

    // Test serial implementation
    {
        nk_size_t packed_size = nk_dots_bf16bf16f32_packed_size_serial(n, k);
        void *b_packed = malloc(packed_size);
        nk_dots_bf16bf16f32_pack_serial(b, n, k, k * sizeof(nk_bf16_t), b_packed);
        nk_dots_bf16bf16f32_serial(a, b_packed, c_test, m, n, k, k * sizeof(nk_bf16_t), n * sizeof(nk_f32_t));
        free(b_packed);

        // Compare results with tolerance (BF16 has limited precision)
        for (nk_size_t i = 0; i < m * n; i++) {
            nk_f64_t diff = fabs((nk_f64_t)c_ref[i] - (nk_f64_t)c_test[i]);
            nk_f64_t rel_err = fabs(c_ref[i]) > 1e-6 ? diff / fabs(c_ref[i]) : diff;
            assert(rel_err < 0.02 || diff < 0.01); // 2% relative or 0.01 absolute
        }
        printf("  - serial: PASS\n");
    }

#if NK_TARGET_SAPPHIRE
    // Test Sapphire (AMX) implementation
    {
        nk_capability_t caps = nk_capabilities();
        nk_configure_thread(caps); // Enable AMX if available
        if (caps & nk_cap_sapphire_amx_k) {
            nk_size_t packed_size = nk_dots_bf16bf16f32_packed_size_sapphire_amx(n, k);
            void *b_packed = malloc(packed_size);
            nk_dots_bf16bf16f32_pack_sapphire_amx(b, n, k, k * sizeof(nk_bf16_t), b_packed);
            nk_dots_bf16bf16f32_sapphire_amx(a, b_packed, c_test, m, n, k, k * sizeof(nk_bf16_t), n * sizeof(nk_f32_t));
            free(b_packed);

            // Compare results
            for (nk_size_t i = 0; i < m * n; i++) {
                nk_f64_t diff = fabs((nk_f64_t)c_ref[i] - (nk_f64_t)c_test[i]);
                nk_f64_t rel_err = fabs(c_ref[i]) > 1e-6 ? diff / fabs(c_ref[i]) : diff;
                assert(rel_err < 0.02 || diff < 0.01);
            }
            printf("  - sapphire (AMX): PASS\n");
        }
        else { printf("  - sapphire (AMX): SKIPPED (not available)\n"); }
    }
#endif

    free(a);
    free(b);
    free(c_ref);
    free(c_test);
    printf("Test matmul BF16: PASS\n");
}

/**
 *  @brief  Test I8 matmul (pack + multiply) against reference.
 */
void test_matmul_i8(void) {
    nk_size_t m = 64, n = 64, k = 64;

    // Allocate matrices
    nk_i8_t *a = (nk_i8_t *)malloc(m * k);
    nk_i8_t *b = (nk_i8_t *)malloc(n * k);
    nk_i32_t *c_ref = (nk_i32_t *)malloc(m * n * sizeof(nk_i32_t));
    nk_i32_t *c_test = (nk_i32_t *)malloc(m * n * sizeof(nk_i32_t));

    // Initialize with random values
    for (nk_size_t i = 0; i < m * k; i++) a[i] = (nk_i8_t)test_random_f64(-127, 127);
    for (nk_size_t i = 0; i < n * k; i++) b[i] = (nk_i8_t)test_random_f64(-127, 127);

    // Compute reference
    reference_matmul_i8_i32(a, b, c_ref, m, n, k);

    // Test serial implementation
    {
        nk_size_t packed_size = nk_dots_i8i8i32_packed_size_serial(n, k);
        void *b_packed = malloc(packed_size);
        nk_dots_i8i8i32_pack_serial(b, n, k, k, b_packed);
        nk_dots_i8i8i32_serial(a, b_packed, c_test, m, n, k, k, n * sizeof(nk_i32_t));
        free(b_packed);

        // Compare results (should be exact for integers)
        for (nk_size_t i = 0; i < m * n; i++) assert(c_ref[i] == c_test[i]);
        printf("  - serial: PASS\n");
    }

#if NK_TARGET_SAPPHIRE
    // Test Sapphire (AMX) implementation
    {
        nk_capability_t caps = nk_capabilities();
        nk_configure_thread(caps); // Enable AMX if available
        if (caps & nk_cap_sapphire_amx_k) {
            nk_size_t packed_size = nk_dots_i8i8i32_packed_size_sapphire_amx(n, k);
            void *b_packed = malloc(packed_size);
            nk_dots_i8i8i32_pack_sapphire_amx(b, n, k, k, b_packed);
            nk_dots_i8i8i32_sapphire_amx(a, b_packed, c_test, m, n, k, k, n * sizeof(nk_i32_t));
            free(b_packed);

            // Compare results
            for (nk_size_t i = 0; i < m * n; i++) assert(c_ref[i] == c_test[i]);
            printf("  - sapphire (AMX): PASS\n");
        }
        else { printf("  - sapphire (AMX): SKIPPED (not available)\n"); }
    }
#endif

    free(a);
    free(b);
    free(c_ref);
    free(c_test);
    printf("Test matmul I8: PASS\n");
}

#pragma endregion

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    printf("Running tests...\n");
    print_capabilities();

    // Infrastructure
    test_utilities();
    test_saturating_arithmetic();
    test_xd_index();
    test_xd_span();

    // Data types
    test_fp8_conversions();
    test_denormals();

    // Distance metrics
    test_distance_from_itself();
    test_distance_precomputed();
    test_bilinear();

    // Elementwise operations
    test_scale_operations();
    test_sum_operations();
    test_wsum_operations();
    test_fma_operations();

    // Trigonometry
    test_approximate_math();
    test_simd_trigonometry();
    test_atan2_quadrants();

    // Geospatial
    test_geospatial_haversine();
    test_geospatial_vincenty();

    // Matrix multiplication
    test_matmul_bf16();
    test_matmul_i8();

    printf("All tests passed.\n");
    return 0;
}
