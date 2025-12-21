/**
 *  @file   test.c
 *  @brief  Test focusing only on the simplest functionality.
 */

#include <assert.h> // `assert`
#include <math.h>   // `sqrtf`
#include <stdio.h>  // `printf`

#define SIMSIMD_NATIVE_F16 0
#define SIMSIMD_NATIVE_BF16 0
#include <simsimd/simsimd.h>

/**
 *  @brief  Logs CPU capabilities supported by the current build (compile-time) and runtime.
 */
void print_capabilities(void) {
    simsimd_capability_t runtime_caps = simsimd_capabilities();

    // Log supported functionality
    char const *flags[2] = {"false", "true"};
    printf("Benchmarking Similarity Measures\n");
    printf("- Compiler used native F16: %s\n", flags[SIMSIMD_NATIVE_F16]);
    printf("- Compiler used native BF16: %s\n", flags[SIMSIMD_NATIVE_BF16]);
    printf("\n");
    printf("Compile-time settings:\n");
    printf("- Arm NEON support enabled: %s\n", flags[SIMSIMD_TARGET_NEON]);
    printf("- Arm SVE support enabled: %s\n", flags[SIMSIMD_TARGET_SVE]);
    printf("- Arm SVE2 support enabled: %s\n", flags[SIMSIMD_TARGET_SVE2]);
    printf("- x86 Haswell support enabled: %s\n", flags[SIMSIMD_TARGET_HASWELL]);
    printf("- x86 Skylake support enabled: %s\n", flags[SIMSIMD_TARGET_SKYLAKE]);
    printf("- x86 Ice Lake support enabled: %s\n", flags[SIMSIMD_TARGET_ICE]);
    printf("- x86 Genoa support enabled: %s\n", flags[SIMSIMD_TARGET_GENOA]);
    printf("- x86 Sapphire Rapids support enabled: %s\n", flags[SIMSIMD_TARGET_SAPPHIRE]);
    printf("- x86 Turin support enabled: %s\n", flags[SIMSIMD_TARGET_TURIN]);
    printf("- x86 Sierra Forest support enabled: %s\n", flags[SIMSIMD_TARGET_SIERRA]);
    printf("\n");
    printf("Run-time settings:\n");
    printf("- Arm NEON support enabled: %s\n", flags[(runtime_caps & simsimd_cap_neon_k) != 0]);
    printf("- Arm NEON F16 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_neon_f16_k) != 0]);
    printf("- Arm NEON BF16 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_neon_bf16_k) != 0]);
    printf("- Arm NEON I8 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_neon_i8_k) != 0]);
    printf("- Arm SVE support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sve_k) != 0]);
    printf("- Arm SVE F16 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sve_f16_k) != 0]);
    printf("- Arm SVE BF16 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sve_bf16_k) != 0]);
    printf("- Arm SVE I8 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sve_i8_k) != 0]);
    printf("- Arm SVE2 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sve2_k) != 0]);
    printf("- x86 Haswell support enabled: %s\n", flags[(runtime_caps & simsimd_cap_haswell_k) != 0]);
    printf("- x86 Skylake support enabled: %s\n", flags[(runtime_caps & simsimd_cap_skylake_k) != 0]);
    printf("- x86 Ice Lake support enabled: %s\n", flags[(runtime_caps & simsimd_cap_ice_k) != 0]);
    printf("- x86 Genoa support enabled: %s\n", flags[(runtime_caps & simsimd_cap_genoa_k) != 0]);
    printf("- x86 Sapphire Rapids support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sapphire_k) != 0]);
    printf("- x86 Turin support enabled: %s\n", flags[(runtime_caps & simsimd_cap_turin_k) != 0]);
    printf("\n");
}

/**
 *  @brief  A trivial test that checks if the utility functions return the expected values.
 */
void test_utilities(void) {
    simsimd_capability_t capabilities = simsimd_capabilities();

    int uses_neon = simsimd_uses_neon();
    int uses_sve = simsimd_uses_sve();
    int uses_haswell = simsimd_uses_haswell();
    int uses_skylake = simsimd_uses_skylake();
    int uses_ice = simsimd_uses_ice();
    int uses_genoa = simsimd_uses_genoa();
    int uses_sapphire = simsimd_uses_sapphire();
    int uses_turin = simsimd_uses_turin();
    int uses_sierra = simsimd_uses_sierra();

    assert(uses_neon == ((capabilities & simsimd_cap_neon_k) != 0));
    assert(uses_sve == ((capabilities & simsimd_cap_sve_k) != 0));
    assert(uses_haswell == ((capabilities & simsimd_cap_haswell_k) != 0));
    assert(uses_skylake == ((capabilities & simsimd_cap_skylake_k) != 0));
    assert(uses_ice == ((capabilities & simsimd_cap_ice_k) != 0));
    assert(uses_genoa == ((capabilities & simsimd_cap_genoa_k) != 0));
    assert(uses_sapphire == ((capabilities & simsimd_cap_sapphire_k) != 0));
    assert(uses_turin == ((capabilities & simsimd_cap_turin_k) != 0));
    assert(uses_sierra == ((capabilities & simsimd_cap_sierra_k) != 0));
}

/**
 *  @brief  A trivial test for internal saturated arithmic function.
 */
void test_saturating_arithmetic(void) {
    // Test cases for addition functions
    simsimd_u8_t u8_a = 200, u8_b = 100, u8_r;
    _simsimd_u8_sadd(&u8_a, &u8_b, &u8_r);
    assert(u8_r == 255); // Overflow case for u8

    simsimd_i8_t i8_a = 100, i8_b = 40, i8_r;
    _simsimd_i8_sadd(&i8_a, &i8_b, &i8_r);
    assert(i8_r == 127); // Overflow case for i8

    simsimd_i8_t i8_under_a = -100, i8_under_b = -40;
    _simsimd_i8_sadd(&i8_under_a, &i8_under_b, &i8_r);
    assert(i8_r == -128); // Underflow case for i8

    simsimd_u16_t u16_a = 50000, u16_b = 20000, u16_r;
    _simsimd_u16_sadd(&u16_a, &u16_b, &u16_r);
    assert(u16_r == 65535); // Overflow case for u16

    simsimd_i16_t i16_a = 30000, i16_b = 10000, i16_r;
    _simsimd_i16_sadd(&i16_a, &i16_b, &i16_r);
    assert(i16_r == 32767); // Overflow case for i16

    simsimd_i16_t i16_under_a = -20000, i16_under_b = -15000;
    _simsimd_i16_sadd(&i16_under_a, &i16_under_b, &i16_r);
    assert(i16_r == -32768); // Underflow case for i16

    // Test cases for multiplication functions
    simsimd_u8_t u8_m_a = 20, u8_m_b = 20, u8_m_r;
    _simsimd_u8_smul(&u8_m_a, &u8_m_b, &u8_m_r);
    assert(u8_m_r == 255); // Overflow case for u8 multiplication

    simsimd_i8_t i8_m_a = 10, i8_m_b = -13, i8_m_r;
    _simsimd_i8_smul(&i8_m_a, &i8_m_b, &i8_m_r);
    assert(i8_m_r == -128); // Underflow case for i8 multiplication

    simsimd_i8_t i8_m_under_a = -100, i8_m_under_b = 2;
    _simsimd_i8_smul(&i8_m_under_a, &i8_m_under_b, &i8_m_r);
    assert(i8_m_r == -128); // Underflow case for i8 multiplication

    simsimd_u16_t u16_m_a = 300, u16_m_b = 300, u16_m_r;
    _simsimd_u16_smul(&u16_m_a, &u16_m_b, &u16_m_r);
    assert(u16_m_r == 65535); // Overflow case for u16 multiplication

    simsimd_i16_t i16_m_a = 200, i16_m_b = 300, i16_m_r;
    _simsimd_i16_smul(&i16_m_a, &i16_m_b, &i16_m_r);
    assert(i16_m_r == 32767); // Overflow case for i16 multiplication

    simsimd_i16_t i16_m_under_a = -200, i16_m_under_b = 300;
    _simsimd_i16_smul(&i16_m_under_a, &i16_m_under_b, &i16_m_r);
    assert(i16_m_r == -32768); // Underflow case for i16 multiplication

    // Normal cases without overflow
    simsimd_u8_t u8_n_a = 20, u8_n_b = 15, u8_n_r;
    _simsimd_u8_sadd(&u8_n_a, &u8_n_b, &u8_n_r);
    assert(u8_n_r == 35);

    simsimd_i8_t i8_n_a = -10, i8_n_b = 20, i8_n_r;
    _simsimd_i8_sadd(&i8_n_a, &i8_n_b, &i8_n_r);
    assert(i8_n_r == 10);

    // Floating-point cases
    simsimd_f32_t f32_a = 1.5f, f32_b = 2.5f, f32_r;
    _simsimd_f32_sadd(&f32_a, &f32_b, &f32_r);
    assert(f32_r == 4.0f); // Normal addition for f32

    simsimd_f32_t f32_m_a = 1.5f, f32_m_b = 2.0f;
    _simsimd_f32_smul(&f32_m_a, &f32_m_b, &f32_r);
    assert(f32_r == 3.0f); // Normal multiplication for f32

    printf("Test saturating arithmetic: PASS\n");
}

/**
 *  @brief  Validating N-Dimensional indexing utilities.
 */
void test_xd_index(void) {
    simsimd_size_t shape[SIMSIMD_NDARRAY_MAX_RANK];
    simsimd_ssize_t strides[SIMSIMD_NDARRAY_MAX_RANK];
    simsimd_xd_index_t xd_index;
    simsimd_ssize_t linear_byte_offset;

    // 1D array
    shape[0] = 10;
    strides[0] = 1 * sizeof(simsimd_u8_t);
    simsimd_xd_index_init(&xd_index);
    for (simsimd_size_t i = 0; i < 10; i++) {
        assert(xd_index.byte_offset == i * sizeof(simsimd_u8_t));
        assert(xd_index.coordinates[0] == i);
        assert(simsimd_xd_index_linearize(shape, strides, 1, &xd_index.coordinates[0], &linear_byte_offset));
        assert(linear_byte_offset == i * sizeof(simsimd_u8_t));
        assert(simsimd_xd_index_next(shape, strides, 1, &xd_index.coordinates[0], &xd_index.byte_offset) == (i < 9));
    }

    // 2D array
    shape[0] = 10, shape[1] = 5;
    strides[0] = 5 * sizeof(simsimd_u8_t), strides[1] = 1 * sizeof(simsimd_u8_t);
    simsimd_xd_index_init(&xd_index);
    for (simsimd_size_t i = 0; i < 10; i++) {
        for (simsimd_size_t j = 0; j < 5; j++) {
            assert(xd_index.byte_offset == (i * 5 + j) * sizeof(simsimd_u8_t));
            assert(xd_index.coordinates[0] == i);
            assert(xd_index.coordinates[1] == j);
            assert(simsimd_xd_index_linearize(shape, strides, 2, &xd_index.coordinates[0], &linear_byte_offset));
            assert(linear_byte_offset == (i * 5 + j) * sizeof(simsimd_u8_t));
            assert(simsimd_xd_index_next(shape, strides, 2, &xd_index.coordinates[0], &xd_index.byte_offset) ==
                   (i != 9 || j != 4));
        }
    }

    // 2D array of complex numbers, taking only the real part
    shape[0] = 10, shape[1] = 5;
    strides[0] = 10 * sizeof(simsimd_u8_t), strides[1] = 2 * sizeof(simsimd_u8_t);
    simsimd_xd_index_init(&xd_index);
    for (simsimd_size_t i = 0; i < 10; i++) {
        for (simsimd_size_t j = 0; j < 5; j++) {
            assert(xd_index.byte_offset == (i * 5 + j) * 2 * sizeof(simsimd_u8_t));
            assert(xd_index.coordinates[0] == i);
            assert(xd_index.coordinates[1] == j);
            assert(simsimd_xd_index_linearize(shape, strides, 2, &xd_index.coordinates[0], &linear_byte_offset));
            assert(linear_byte_offset == (i * 5 + j) * 2 * sizeof(simsimd_u8_t));
            assert(simsimd_xd_index_next(shape, strides, 2, &xd_index.coordinates[0], &xd_index.byte_offset) ==
                   (i != 9 || j != 4));
        }
    }

    // 3D array with different strides at every level
    // At each level it should be at least as big as the smaller level stride
    // multiplied by its size, otherwise we interleave the data.
    shape[0] = 10, shape[1] = 5, shape[2] = 3;
    strides[0] = 41 * sizeof(simsimd_u8_t), strides[1] = 7 * sizeof(simsimd_u8_t),
    strides[2] = 2 * sizeof(simsimd_u8_t);
    simsimd_xd_index_init(&xd_index);
    for (simsimd_size_t i = 0; i < 10; i++) {
        for (simsimd_size_t j = 0; j < 5; j++) {
            for (simsimd_size_t k = 0; k < 3; k++) {
                assert(xd_index.byte_offset == (i * strides[0] + j * strides[1] + k * strides[2]));
                assert(xd_index.coordinates[0] == i);
                assert(xd_index.coordinates[1] == j);
                assert(xd_index.coordinates[2] == k);
                assert(simsimd_xd_index_linearize(shape, strides, 3, &xd_index.coordinates[0], &linear_byte_offset));
                assert(linear_byte_offset == (i * strides[0] + j * strides[1] + k * strides[2]));
                assert(simsimd_xd_index_next(shape, strides, 3, &xd_index.coordinates[0], &xd_index.byte_offset) ==
                       (i != 9 || j != 4 || k != 2));
            }
        }
    }

    // Populated 3D array with different strides at every level
    {
        simsimd_f32_t tensor[11][43][7];
        // Fill tensor with values
        for (simsimd_size_t i = 0; i < 11; i++) {
            for (simsimd_size_t j = 0; j < 43; j++)
                for (simsimd_size_t k = 0; k < 7; k++) tensor[i][j][k] = i * 10000 + j * 100 + k * 1;
        }
        // Accumulate a slice: tensor[1:9:2, 2:42:4, 1:5:3] ~ 4 channels, 10 rows, 2 columns
        simsimd_xd_index_init(&xd_index);
        shape[0] = _simsimd_divide_ceil(9 - 1, 2);
        shape[1] = _simsimd_divide_ceil(42 - 2, 4);
        shape[2] = _simsimd_divide_ceil(5 - 1, 3);
        strides[0] = 43 * 7 * sizeof(simsimd_f32_t) * 2; // Physical size of 2 channels
        strides[1] = 7 * sizeof(simsimd_f32_t) * 4;      // Physical size of 4 rows
        strides[2] = 3 * sizeof(simsimd_f32_t);          // Physical size of 3 columns
        // Accumulate using native indexing
        simsimd_f32_t sum_native = 0;
        for (simsimd_size_t i = 1; i < 9; i += 2) {
            for (simsimd_size_t j = 2; j < 42; j += 4) {
                for (simsimd_size_t k = 1; k < 5; k += 3) { //
                    sum_native += tensor[i][j][k];
                }
            }
        }
        // Accumulate using our `simsimd_xd_index_t` iterator
        simsimd_f32_t sum_with_xd_index = 0;
        simsimd_f32_t sum_native_running = 0;
        for (simsimd_size_t i = 1; i < 9; i += 2) {
            for (simsimd_size_t j = 2; j < 42; j += 4) {
                for (simsimd_size_t k = 1; k < 5; k += 3) {
                    simsimd_size_t const expected_global_offset = //
                        ((i - 1) / 2) * shape[1] * shape[2] +     //
                        ((j - 2) / 4) * shape[2] +                //
                        ((k - 1) / 3);                            //
                    simsimd_f32_t const entry_native = tensor[i][j][k];
                    simsimd_f32_t const entry_from_byte_offset =
                        *(simsimd_f32_t *)_simsimd_advance_by_bytes(&tensor[1][2][1], xd_index.byte_offset);
                    simsimd_f32_t const entry_from_coordinate = tensor //
                        [xd_index.coordinates[0] * 2 + 1]              //
                        [xd_index.coordinates[1] * 4 + 2]              //
                        [xd_index.coordinates[2] * 3 + 1];
                    assert(entry_native == entry_from_byte_offset);
                    assert(entry_native == entry_from_coordinate);
                    sum_with_xd_index += entry_from_byte_offset;
                    sum_native_running += entry_native;
                    assert(sum_native_running == sum_with_xd_index);
                    simsimd_xd_index_next(shape, strides, 3, &xd_index.coordinates[0], &xd_index.byte_offset);
                }
            }
        }
    }

    printf("Test xd_index: PASS\n");
}

/**
 *  @brief  Test simsimd_xd_span_init function.
 */
void test_xd_span(void) {
    simsimd_xd_span_t xd_span;

    // Initialize the span
    simsimd_xd_span_init(&xd_span);

    // Verify all extents are zero
    for (simsimd_size_t i = 0; i < SIMSIMD_NDARRAY_MAX_RANK; i++) {
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

/**
 *  @brief  Goes through all possible `f32` values in a relevant range, computing
 */
void test_approximate_math(void) {

    typedef struct error_aggregator {
        simsimd_f64_t absolute_error;
        simsimd_f64_t relative_error;
        simsimd_f64_t max_error;
    } error_aggregator;

    error_aggregator f32_cos_errors = {0, 0, 0}, f32_sin_errors = {0, 0, 0};
    error_aggregator f32_atan_errors = {0, 0, 0}, f32_atan2_errors = {0, 0, 0};
    error_aggregator f64_cos_errors = {0, 0, 0}, f64_sin_errors = {0, 0, 0};
    error_aggregator f64_atan_errors = {0, 0, 0}, f64_atan2_errors = {0, 0, 0};

    simsimd_f32_t const range_min = -3.14159265358979323846f * 2;
    simsimd_f32_t const range_max = 3.14159265358979323846f * 2;

    // Test all possible values of f32 within ranges: [-π, -1], [-1, -0], [0, 1], [1, π].
    simsimd_size_t const step = 1;
    simsimd_size_t const count_tests = 0xFFFFFFFFull / step;
    union {
        simsimd_f32_t f32;
        simsimd_u32_t u32;
    } x;

    // Run separate loops for every operation
    for (x.u32 = 0; (x.u32 + step) <= 0xFFFFFFFFull; x.u32 += step) {
        if (x.f32 < range_min || x.f32 > range_max) continue;
        simsimd_f32_t f32_sin_baseline = sinf(x.f32);
        simsimd_f32_t f32_sin_approx = simsimd_f32_sin(x.f32);
        simsimd_f32_t f32_sin_diff = fabsf(f32_sin_baseline - f32_sin_approx);
        simsimd_f32_t f32_sin_max = fmaxf(fabsf(f32_sin_baseline), fabsf(f32_sin_approx));
        f32_sin_errors.absolute_error += f32_sin_diff;
        f32_sin_errors.relative_error += f32_sin_max != 0 ? f32_sin_diff / f32_sin_max : 0;
        f32_sin_errors.max_error = fmax(f32_sin_errors.max_error, f32_sin_diff);
    }
    printf("f32 sin: <error>= %f, up to %f\n", f32_sin_errors.absolute_error / count_tests, f32_sin_errors.max_error);
    for (x.u32 = 0; (x.u32 + step) <= 0xFFFFFFFFull; x.u32 += step) {
        if (x.f32 < range_min || x.f32 > range_max) continue;
        simsimd_f32_t f32_cos_baseline = cosf(x.f32);
        simsimd_f32_t f32_cos_approx = simsimd_f32_cos(x.f32);
        simsimd_f32_t f32_cos_diff = fabsf(f32_cos_baseline - f32_cos_approx);
        simsimd_f32_t f32_cos_max = fmaxf(fabsf(f32_cos_baseline), fabsf(f32_cos_approx));
        f32_cos_errors.absolute_error += f32_cos_diff;
        f32_cos_errors.relative_error += f32_cos_max != 0 ? f32_cos_diff / f32_cos_max : 0;
        f32_cos_errors.max_error = fmax(f32_cos_errors.max_error, f32_cos_diff);
    }
    printf("f32 cos: <error>= %f, up to %f\n", f32_cos_errors.absolute_error / count_tests, f32_cos_errors.max_error);
    for (x.u32 = 0; (x.u32 + step) <= 0xFFFFFFFFull; x.u32 += step) {
        if (x.f32 < range_min || x.f32 > range_max) continue;
        simsimd_f32_t f32_atan_baseline = atanf(x.f32);
        simsimd_f32_t f32_atan_approx = simsimd_f32_atan(x.f32);
        simsimd_f32_t f32_atan_diff = fabs(f32_atan_baseline - f32_atan_approx);
        simsimd_f32_t f32_atan_max = fmax(fabs(f32_atan_baseline), fabs(f32_atan_approx));
        f32_atan_errors.absolute_error += f32_atan_diff;
        f32_atan_errors.relative_error += f32_atan_max != 0 ? f32_atan_diff / f32_atan_max : 0;
        f32_atan_errors.max_error = fmax(f32_atan_errors.max_error, f32_atan_diff);
    }
    printf("f32 atan: <error>= %f, up to %f\n", f32_atan_errors.absolute_error / count_tests,
           f32_atan_errors.max_error);
    for (x.u32 = 0; (x.u32 + step) <= 0xFFFFFFFFull; x.u32 += step) {
        if (x.f32 < range_min || x.f32 > range_max) continue;
        simsimd_f32_t f32_atan2_baseline = atan2f(x.f32, (1 - x.f32));
        simsimd_f32_t f32_atan2_approx = simsimd_f32_atan2(x.f32, (1 - x.f32));
        simsimd_f32_t f32_atan2_diff = fabs(f32_atan2_baseline - f32_atan2_approx);
        simsimd_f32_t f32_atan2_max = fmax(fabs(f32_atan2_baseline), fabs(f32_atan2_approx));
        f32_atan2_errors.absolute_error += f32_atan2_diff;
        f32_atan2_errors.relative_error += f32_atan2_max != 0 ? f32_atan2_diff / f32_atan2_max : 0;
        f32_atan2_errors.max_error = fmax(f32_atan2_errors.max_error, f32_atan2_diff);
    }
    printf("f32 atan2: <error>= %f, up to %f\n", f32_atan2_errors.absolute_error / count_tests,
           f32_atan2_errors.max_error);
    for (x.u32 = 0; (x.u32 + step) <= 0xFFFFFFFFull; x.u32 += step) {
        if (x.f32 < range_min || x.f32 > range_max) continue;
        simsimd_f64_t f64_sin_baseline = sin(x.f32);
        simsimd_f64_t f64_sin_approx = simsimd_f64_sin(x.f32);
        simsimd_f64_t f64_sin_diff = fabs(f64_sin_baseline - f64_sin_approx);
        simsimd_f64_t f64_sin_max = fmax(fabs(f64_sin_baseline), fabs(f64_sin_approx));
        f64_sin_errors.absolute_error += f64_sin_diff;
        f64_sin_errors.relative_error += f64_sin_max != 0 ? f64_sin_diff / f64_sin_max : 0;
        f64_sin_errors.max_error = fmax(f64_sin_errors.max_error, f64_sin_diff);
    }
    printf("f64 sin: <error>= %f, up to %f\n", f64_sin_errors.absolute_error / count_tests, f64_sin_errors.max_error);
    for (x.u32 = 0; (x.u32 + step) <= 0xFFFFFFFFull; x.u32 += step) {
        if (x.f32 < range_min || x.f32 > range_max) continue;
        simsimd_f64_t f64_cos_baseline = cos(x.f32);
        simsimd_f64_t f64_cos_approx = simsimd_f64_cos(x.f32);
        simsimd_f64_t f64_cos_diff = fabs(f64_cos_baseline - f64_cos_approx);
        simsimd_f64_t f64_cos_max = fmax(fabs(f64_cos_baseline), fabs(f64_cos_approx));
        f64_cos_errors.absolute_error += f64_cos_diff;
        f64_cos_errors.relative_error += f64_cos_max != 0 ? f64_cos_diff / f64_cos_max : 0;
        f64_cos_errors.max_error = fmax(f64_cos_errors.max_error, f64_cos_diff);
    }
    printf("f64 cos: <error>= %f, up to %f\n", f64_cos_errors.absolute_error / count_tests, f64_cos_errors.max_error);
    for (x.u32 = 0; (x.u32 + step) <= 0xFFFFFFFFull; x.u32 += step) {
        if (x.f32 < range_min || x.f32 > range_max) continue;
        simsimd_f64_t f64_atan_baseline = atan(x.f32);
        simsimd_f64_t f64_atan_approx = simsimd_f64_atan(x.f32);
        simsimd_f64_t f64_atan_diff = fabs(f64_atan_baseline - f64_atan_approx);
        simsimd_f64_t f64_atan_max = fmax(fabs(f64_atan_baseline), fabs(f64_atan_approx));
        f64_atan_errors.absolute_error += f64_atan_diff;
        f64_atan_errors.relative_error += f64_atan_max != 0 ? f64_atan_diff / f64_atan_max : 0;
        f64_atan_errors.max_error = fmax(f64_atan_errors.max_error, f64_atan_diff);
    }
    printf("f64 atan: <error>= %f, up to %f\n", f64_atan_errors.absolute_error / count_tests,
           f64_atan_errors.max_error);
    for (x.u32 = 0; (x.u32 + step) <= 0xFFFFFFFFull; x.u32 += step) {
        if (x.f32 < range_min || x.f32 > range_max) continue;
        simsimd_f64_t f64_atan2_baseline = atan2(x.f32, (1 - x.f32));
        simsimd_f64_t f64_atan2_approx = simsimd_f64_atan2(x.f32, (1 - x.f32));
        simsimd_f64_t f64_atan2_diff = fabs(f64_atan2_baseline - f64_atan2_approx);
        simsimd_f64_t f64_atan2_max = fmax(fabs(f64_atan2_baseline), fabs(f64_atan2_approx));
        f64_atan2_errors.absolute_error += f64_atan2_diff;
        f64_atan2_errors.relative_error += f64_atan2_max != 0 ? f64_atan2_diff / f64_atan2_max : 0;
        f64_atan2_errors.max_error = fmax(f64_atan2_errors.max_error, f64_atan2_diff);
    }
    printf("f64 atan2: <error>= %f, up to %f\n", f64_atan2_errors.absolute_error / count_tests,
           f64_atan2_errors.max_error);
}

/**
 *  @brief  Test FP8 E4M3 conversion functions.
 */
void test_fp8_conversions(void) {
    printf("Testing FP8 E4M3 conversions...\n");

    // Test conversion of common values
    simsimd_f32_t test_values[] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, 10.0f, -10.0f, 100.0f};
    simsimd_size_t num_tests = sizeof(test_values) / sizeof(test_values[0]);

    for (simsimd_size_t i = 0; i < num_tests; i++) {
        simsimd_f32_t original = test_values[i];
        simsimd_e4m3_t e4m3_value;
        simsimd_f32_t reconstructed;

        // Convert f32 to e4m3
        simsimd_f32_to_e4m3(&original, &e4m3_value);

        // Convert e4m3 back to f32
        simsimd_e4m3_to_f32(&e4m3_value, &reconstructed);

        // Check if sign is preserved
        if (original != 0.0f) {
            assert((original > 0.0f && reconstructed >= 0.0f) || (original < 0.0f && reconstructed <= 0.0f));
        }

        // For small values, check relative error
        if (fabsf(original) > 0.1f && fabsf(original) < 100.0f) {
            simsimd_f32_t relative_error = fabsf((original - reconstructed) / original);
            // FP8 has limited precision, so we allow up to 20% relative error for this test
            assert(relative_error < 0.2f);
        }
    }

    // Test round-trip conversion for zero
    {
        simsimd_f32_t zero = 0.0f;
        simsimd_e4m3_t e4m3_zero;
        simsimd_f32_to_e4m3(&zero, &e4m3_zero);
        simsimd_f32_t result;
        simsimd_e4m3_to_f32(&e4m3_zero, &result);
        assert(result == 0.0f);
    }

    printf("Test FP8 conversions: PASS\n");
}

/**
 *  @brief  Test scale operation for various data types.
 */
void test_scale_operations(void) {
    printf("Testing scale operations...\n");

    // Test f32 scale
    {
        simsimd_f32_t input[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        simsimd_f32_t result[8];
        simsimd_distance_t alpha = 2.0;
        simsimd_distance_t beta = 1.0;

        simsimd_scale_f32(input, 8, alpha, beta, result);

        // Check: result = alpha * input + beta
        for (simsimd_size_t i = 0; i < 8; i++) {
            simsimd_f32_t expected = alpha * input[i] + beta;
            assert(fabsf(result[i] - expected) < 0.001f);
        }
    }

    // Test f64 scale
    {
        simsimd_f64_t input[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        simsimd_f64_t result[8];
        simsimd_distance_t alpha = 3.0;
        simsimd_distance_t beta = 0.5;

        simsimd_scale_f64(input, 8, alpha, beta, result);

        for (simsimd_size_t i = 0; i < 8; i++) {
            simsimd_f64_t expected = alpha * input[i] + beta;
            assert(fabs(result[i] - expected) < 0.001);
        }
    }

    // Test i8 scale
    {
        simsimd_i8_t input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        simsimd_i8_t result[8];
        simsimd_distance_t alpha = 2.0;
        simsimd_distance_t beta = 1.0;

        simsimd_scale_i8(input, 8, alpha, beta, result);

        // Check results (with saturation in mind)
        for (simsimd_size_t i = 0; i < 8; i++) {
            simsimd_f32_t expected_f32 = alpha * input[i] + beta;
            simsimd_i8_t expected =
                (simsimd_i8_t)(expected_f32 > 127 ? 127 : (expected_f32 < -128 ? -128 : expected_f32));
            assert(result[i] == expected);
        }
    }

    // Test u8 scale
    {
        simsimd_u8_t input[8] = {10, 20, 30, 40, 50, 60, 70, 80};
        simsimd_u8_t result[8];
        simsimd_distance_t alpha = 1.5;
        simsimd_distance_t beta = 5.0;

        simsimd_scale_u8(input, 8, alpha, beta, result);

        // Check results (with saturation in mind)
        for (simsimd_size_t i = 0; i < 8; i++) {
            simsimd_f32_t expected_f32 = alpha * input[i] + beta;
            simsimd_u8_t expected = (simsimd_u8_t)(expected_f32 > 255 ? 255 : (expected_f32 < 0 ? 0 : expected_f32));
            assert(result[i] == expected);
        }
    }

    printf("Test scale operations: PASS\n");
}

/**
 *  @brief  Test sum operation for various data types.
 */
void test_sum_operations(void) {
    printf("Testing sum operations...\n");

    // Test f32 sum
    {
        simsimd_f32_t a[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        simsimd_f32_t b[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        simsimd_f32_t result[8];

        simsimd_sum_f32(a, b, 8, result);

        for (simsimd_size_t i = 0; i < 8; i++) { assert(fabsf(result[i] - (a[i] + b[i])) < 0.001f); }
    }

    // Test f64 sum
    {
        simsimd_f64_t a[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        simsimd_f64_t b[8] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5};
        simsimd_f64_t result[8];

        simsimd_sum_f64(a, b, 8, result);

        for (simsimd_size_t i = 0; i < 8; i++) { assert(fabs(result[i] - (a[i] + b[i])) < 0.001); }
    }

    // Test i8 sum
    {
        simsimd_i8_t a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        simsimd_i8_t b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
        simsimd_i8_t result[8];

        simsimd_sum_i8(a, b, 8, result);

        for (simsimd_size_t i = 0; i < 8; i++) {
            simsimd_i16_t expected_i16 = (simsimd_i16_t)a[i] + (simsimd_i16_t)b[i];
            simsimd_i8_t expected =
                (simsimd_i8_t)(expected_i16 > 127 ? 127 : (expected_i16 < -128 ? -128 : expected_i16));
            assert(result[i] == expected);
        }
    }

    // Test u8 sum
    {
        simsimd_u8_t a[8] = {10, 20, 30, 40, 50, 60, 70, 80};
        simsimd_u8_t b[8] = {5, 15, 25, 35, 45, 55, 65, 75};
        simsimd_u8_t result[8];

        simsimd_sum_u8(a, b, 8, result);

        for (simsimd_size_t i = 0; i < 8; i++) {
            simsimd_u16_t expected_u16 = (simsimd_u16_t)a[i] + (simsimd_u16_t)b[i];
            simsimd_u8_t expected = (simsimd_u8_t)(expected_u16 > 255 ? 255 : expected_u16);
            assert(result[i] == expected);
        }
    }

    printf("Test sum operations: PASS\n");
}

/**
 *  @brief  A trivial test that calls every implemented distance function and their dispatch versions
 *          on vectors A and B, where A and B are equal.
 */
void test_distance_from_itself(void) {
    simsimd_f64_t f64s[1536];
    simsimd_f32_t f32s[1536];
    simsimd_f16_t f16s[1536];
    simsimd_bf16_t bf16s[1536];

    simsimd_f64c_t f64cs[768];
    simsimd_f32c_t f32cs[768];
    simsimd_f16c_t f16cs[768];
    simsimd_bf16c_t bf16cs[768];

    simsimd_i8_t i8s[1536];
    simsimd_u8_t u8s[1536];
    simsimd_b8_t b8s[1536 / 8];     // 8 bits per word
    simsimd_distance_t distance[2]; // For complex dot-products we need two values

    // Cosine distance between two vectors
    simsimd_angular_i8(i8s, i8s, 1536, &distance[0]);
    simsimd_angular_u8(u8s, u8s, 1536, &distance[0]);
    simsimd_angular_f16(f16s, f16s, 1536, &distance[0]);
    simsimd_angular_bf16(bf16s, bf16s, 1536, &distance[0]);
    simsimd_angular_f32(f32s, f32s, 1536, &distance[0]);
    simsimd_angular_f64(f64s, f64s, 1536, &distance[0]);

    // Euclidean distance between two vectors
    simsimd_l2sq_i8(i8s, i8s, 1536, &distance[0]);
    simsimd_l2sq_u8(u8s, u8s, 1536, &distance[0]);
    simsimd_l2sq_f16(f16s, f16s, 1536, &distance[0]);
    simsimd_l2sq_bf16(bf16s, bf16s, 1536, &distance[0]);
    simsimd_l2sq_f32(f32s, f32s, 1536, &distance[0]);
    simsimd_l2sq_f64(f64s, f64s, 1536, &distance[0]);

    // Inner product between two vectors
    simsimd_dot_i8(i8s, i8s, 1536, &distance[0]);
    simsimd_dot_u8(u8s, u8s, 1536, &distance[0]);
    simsimd_dot_f16(f16s, f16s, 1536, &distance[0]);
    simsimd_dot_bf16(bf16s, bf16s, 1536, &distance[0]);
    simsimd_dot_f32(f32s, f32s, 1536, &distance[0]);
    simsimd_dot_f64(f64s, f64s, 1536, &distance[0]);

    // Complex inner product between two vectors
    simsimd_dot_bf16c(bf16cs, bf16cs, 768, &distance[0]);
    simsimd_dot_f16c(f16cs, f16cs, 768, &distance[0]);
    simsimd_dot_f32c(f32cs, f32cs, 768, &distance[0]);
    simsimd_dot_f64c(f64cs, f64cs, 768, &distance[0]);

    // Complex conjugate inner product between two vectors
    simsimd_vdot_bf16c(bf16cs, bf16cs, 768, &distance[0]);
    simsimd_vdot_f16c(f16cs, f16cs, 768, &distance[0]);
    simsimd_vdot_f32c(f32cs, f32cs, 768, &distance[0]);
    simsimd_vdot_f64c(f64cs, f64cs, 768, &distance[0]);

    // Hamming distance between two vectors
    simsimd_hamming_b8(b8s, b8s, 1536 / 8, &distance[0]);

    // Jaccard distance between two vectors
    simsimd_jaccard_b8(b8s, b8s, 1536 / 8, &distance[0]);

    // Jensen-Shannon divergence between two vectors
    simsimd_jsd_f16(f16s, f16s, 1536, &distance[0]);
    simsimd_jsd_bf16(bf16s, bf16s, 1536, &distance[0]);
    simsimd_jsd_f32(f32s, f32s, 1536, &distance[0]);
    simsimd_jsd_f64(f64s, f64s, 1536, &distance[0]);

    // Kullback-Leibler divergence between two vectors
    simsimd_kld_f16(f16s, f16s, 1536, &distance[0]);
    simsimd_kld_bf16(bf16s, bf16s, 1536, &distance[0]);
    simsimd_kld_f32(f32s, f32s, 1536, &distance[0]);
    simsimd_kld_f64(f64s, f64s, 1536, &distance[0]);
}

/**
 *  @brief Test whether denormals are being flushed to zero or not.
 *
 *  We create subnormal float values, do a small computation (multiplication),
 *  and classify the result. If flush-to-zero @b (FTZ) is enabled, the result is
 *  likely zero. Otherwise, you may see another subnormal or normal number.
 */
static void test_denormals(void) {

    // Create two subnormal floats:
    // 1e-40 ~ 1.0 * 10^-40 is typically a subnormal in IEEE-754 single precision
    float subnorm1 = 1e-40f;
    float subnorm2 = 2e-40f;
    float result = subnorm1 * subnorm2; // This might be subnormal, zero, or normal
    int classification = fpclassify(result);
    if (classification == FP_SUBNORMAL) { printf("Denormal test: result is subnormal: %.8g\n", result); }
    else if (result == 0.0f) { printf("Denormal test: result is zero (denormals likely flushed).\n"); }
    else if (classification == FP_NORMAL) { printf("Denormal test: result is normal: %.8g\n", result); }
    else { printf("Denormal test: result has unexpected classification.\n"); }
}

int main(int argc, char **argv) {
    printf("Running tests...\n");
    print_capabilities();
    test_utilities();
    test_saturating_arithmetic();
    test_xd_index();
    test_xd_span();
    test_fp8_conversions();
    test_scale_operations();
    test_sum_operations();
    test_approximate_math();
    test_distance_from_itself();
    test_denormals();
    printf("All tests passed.\n");
    return 0;
}
