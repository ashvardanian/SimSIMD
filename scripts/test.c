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
 *  @brief  Validating N-Dimensional indexing utilities.
 */
void test_ndindex(void) {
    simsimd_size_t shape[SIMSIMD_NDARRAY_MAX_RANK];
    simsimd_size_t strides[SIMSIMD_NDARRAY_MAX_RANK];
    simsimd_ndindex_t ndindex;

    // 1D array
    shape[0] = 10;
    strides[0] = 1 * sizeof(simsimd_u8_t);
    simsimd_ndindex_init(&ndindex);
    for (simsimd_size_t i = 0; i < 10; i++) {
        assert(ndindex.global_offset == i);
        assert(ndindex.byte_offset == i * sizeof(simsimd_u8_t));
        assert(ndindex.coordinate[0] == i);
        assert(simsimd_ndindex_next(&ndindex, 1, shape, strides) == (i < 9));
    }

    // 2D array
    shape[0] = 10, shape[1] = 5;
    strides[0] = 5 * sizeof(simsimd_u8_t), strides[1] = 1 * sizeof(simsimd_u8_t);
    simsimd_ndindex_init(&ndindex);
    for (simsimd_size_t i = 0; i < 10; i++) {
        for (simsimd_size_t j = 0; j < 5; j++) {
            assert(ndindex.global_offset == i * 5 + j);
            assert(ndindex.byte_offset == (i * 5 + j) * sizeof(simsimd_u8_t));
            assert(ndindex.coordinate[0] == i);
            assert(ndindex.coordinate[1] == j);
            assert(simsimd_ndindex_next(&ndindex, 2, shape, strides) == (i != 9 || j != 4));
        }
    }

    // 2D array of complex numbers, taking only the real part
    shape[0] = 10, shape[1] = 5;
    strides[0] = 10 * sizeof(simsimd_u8_t), strides[1] = 2 * sizeof(simsimd_u8_t);
    simsimd_ndindex_init(&ndindex);
    for (simsimd_size_t i = 0; i < 10; i++) {
        for (simsimd_size_t j = 0; j < 5; j++) {
            assert(ndindex.global_offset == i * 5 + j);
            assert(ndindex.byte_offset == (i * 5 + j) * 2 * sizeof(simsimd_u8_t));
            assert(ndindex.coordinate[0] == i);
            assert(ndindex.coordinate[1] == j);
            assert(simsimd_ndindex_next(&ndindex, 2, shape, strides) == (i != 9 || j != 4));
        }
    }
    assert(ndindex.global_offset == 10 * 5);

    // 3D array with different strides at every level
    // At each level it should be at least as big as the smaller level stride
    // multiplied by its size, otherwise we interleave the data.
    shape[0] = 10, shape[1] = 5, shape[2] = 3;
    strides[0] = 41 * sizeof(simsimd_u8_t), strides[1] = 7 * sizeof(simsimd_u8_t),
    strides[2] = 2 * sizeof(simsimd_u8_t);
    simsimd_ndindex_init(&ndindex);
    for (simsimd_size_t i = 0; i < 10; i++) {
        for (simsimd_size_t j = 0; j < 5; j++) {
            for (simsimd_size_t k = 0; k < 3; k++) {
                assert(ndindex.global_offset == i * 15 + j * 3 + k);
                assert(ndindex.byte_offset == (i * strides[0] + j * strides[1] + k * strides[2]));
                assert(ndindex.coordinate[0] == i);
                assert(ndindex.coordinate[1] == j);
                assert(ndindex.coordinate[2] == k);
                assert(simsimd_ndindex_next(&ndindex, 3, shape, strides) == (i != 9 || j != 4 || k != 2));
            }
        }
    }
    assert(ndindex.global_offset == 10 * 5 * 3);

    // Populated 3D array with different strides at every level
    {
        simsimd_f32_t tensor[11][43][7];
        // Fill tensor with values
        for (simsimd_size_t i = 0; i < 11; i++) {
            for (simsimd_size_t j = 0; j < 43; j++)
                for (simsimd_size_t k = 0; k < 7; k++) tensor[i][j][k] = i * 10000 + j * 100 + k * 1;
        }
        // Accumulate a slice: tensor[1:9:2, 2:42:4, 1:5:3] ~ 4 channels, 10 rows, 2 columns
        simsimd_ndindex_init(&ndindex);
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
        // Accumulate using our `simsimd_ndindex_t` iterator
        simsimd_f32_t sum_with_ndindex = 0;
        simsimd_f32_t sum_native_running = 0;
        for (simsimd_size_t i = 1; i < 9; i += 2) {
            for (simsimd_size_t j = 2; j < 42; j += 4) {
                for (simsimd_size_t k = 1; k < 5; k += 3) {
                    simsimd_size_t const expected_global_offset = //
                        ((i - 1) / 2) * shape[1] * shape[2] +     //
                        ((j - 2) / 4) * shape[2] +                //
                        ((k - 1) / 3);                            //
                    assert(ndindex.global_offset == expected_global_offset);
                    simsimd_f32_t const entry_native = tensor[i][j][k];
                    simsimd_f32_t const entry_from_byte_offset =
                        *(simsimd_f32_t *)_simsimd_advance_by_bytes(&tensor[1][2][1], ndindex.byte_offset);
                    simsimd_f32_t const entry_from_coordinate = tensor //
                        [ndindex.coordinate[0] * 2 + 1]                //
                        [ndindex.coordinate[1] * 4 + 2]                //
                        [ndindex.coordinate[2] * 3 + 1];
                    assert(entry_native == entry_from_byte_offset);
                    assert(entry_native == entry_from_coordinate);
                    sum_with_ndindex += entry_from_byte_offset;
                    sum_native_running += entry_native;
                    assert(sum_native_running == sum_with_ndindex);
                    simsimd_ndindex_next(&ndindex, 3, shape, strides);
                }
            }
        }
    }
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
    simsimd_i8_t i8s[1536];
    simsimd_u8_t u8s[1536];
    simsimd_b8_t b8s[1536 / 8];     // 8 bits per word
    simsimd_distance_t distance[2]; // For complex dot-products we need two values

    // Cosine distance between two vectors
    simsimd_cos_i8(i8s, i8s, 1536, &distance[0]);
    simsimd_cos_u8(u8s, u8s, 1536, &distance[0]);
    simsimd_cos_f16(f16s, f16s, 1536, &distance[0]);
    simsimd_cos_bf16(bf16s, bf16s, 1536, &distance[0]);
    simsimd_cos_f32(f32s, f32s, 1536, &distance[0]);
    simsimd_cos_f64(f64s, f64s, 1536, &distance[0]);

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
    simsimd_dot_f16c(f16s, f16s, 1536, &distance[0]);
    simsimd_dot_bf16c(bf16s, bf16s, 1536, &distance[0]);
    simsimd_dot_f32c(f32s, f32s, 1536, &distance[0]);
    simsimd_dot_f64c(f64s, f64s, 1536, &distance[0]);

    // Complex conjugate inner product between two vectors
    simsimd_vdot_f16c(f16s, f16s, 1536, &distance[0]);
    simsimd_vdot_bf16c(bf16s, bf16s, 1536, &distance[0]);
    simsimd_vdot_f32c(f32s, f32s, 1536, &distance[0]);
    simsimd_vdot_f64c(f64s, f64s, 1536, &distance[0]);

    // Hamming distance between two vectors
    simsimd_hamming_b8(b8s, b8s, 1536 / 8, &distance[0]);

    // Jaccard distance between two vectors
    simsimd_jaccard_b8(b8s, b8s, 1536 / 8, &distance[0]);

    // Jensen-Shannon divergence between two vectors
    simsimd_js_f16(f16s, f16s, 1536, &distance[0]);
    simsimd_js_bf16(bf16s, bf16s, 1536, &distance[0]);
    simsimd_js_f32(f32s, f32s, 1536, &distance[0]);
    simsimd_js_f64(f64s, f64s, 1536, &distance[0]);

    // Kullback-Leibler divergence between two vectors
    simsimd_kl_f16(f16s, f16s, 1536, &distance[0]);
    simsimd_kl_bf16(bf16s, bf16s, 1536, &distance[0]);
    simsimd_kl_f32(f32s, f32s, 1536, &distance[0]);
    simsimd_kl_f64(f64s, f64s, 1536, &distance[0]);
}

int main(int argc, char **argv) {
    printf("Running tests...\n");
    print_capabilities();
    test_utilities();
    test_ndindex();
    test_distance_from_itself();
    printf("All tests passed.\n");
    return 0;
}
