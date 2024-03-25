/**
 *  @file   test.c
 *  @brief  Test focusing only on the simplest functionality.
 */

#include <assert.h> // `assert`
#include <math.h>   // `sqrtf`
#include <stdio.h>  // `printf`

#define SIMSIMD_NATIVE_F16 0
#define SIMSIMD_RSQRT(x) (1 / sqrtf(x))
#define SIMSIMD_LOG(x) (logf(x))
#include <simsimd/simsimd.h>

/**
 *  @brief  Logs CPU capabilities supported by the current build (compile-time) and runtime.
 */
void print_capabilities() {
    simsimd_capability_t runtime_caps = simsimd_capabilities();

    // Log supported functionality
    char const* flags[2] = {"false", "true"};
    printf("Benchmarking Similarity Measures\n");
    printf("- Compiler used native F16: %s\n", flags[SIMSIMD_NATIVE_F16]);
    printf("\n");
    printf("Compile-time settings:\n");
    printf("- Arm NEON support enabled: %s\n", flags[SIMSIMD_TARGET_NEON]);
    printf("- Arm SVE support enabled: %s\n", flags[SIMSIMD_TARGET_SVE]);
    printf("- x86 Haswell support enabled: %s\n", flags[SIMSIMD_TARGET_HASWELL]);
    printf("- x86 Skylake support enabled: %s\n", flags[SIMSIMD_TARGET_SKYLAKE]);
    printf("- x86 Ice Lake support enabled: %s\n", flags[SIMSIMD_TARGET_ICE]);
    printf("- x86 Sapphire Rapids support enabled: %s\n", flags[SIMSIMD_TARGET_SAPPHIRE]);
    printf("\n");
    printf("Run-time settings:\n");
    printf("- Arm NEON support enabled: %s\n", flags[(runtime_caps & simsimd_cap_neon_k) != 0]);
    printf("- Arm SVE support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sve_k) != 0]);
    printf("- x86 Haswell support enabled: %s\n", flags[(runtime_caps & simsimd_cap_haswell_k) != 0]);
    printf("- x86 Skylake support enabled: %s\n", flags[(runtime_caps & simsimd_cap_skylake_k) != 0]);
    printf("- x86 Ice Lake support enabled: %s\n", flags[(runtime_caps & simsimd_cap_ice_k) != 0]);
    printf("- x86 Sapphire Rapids support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sapphire_k) != 0]);
    printf("\n");
}

void test_utilities() {
    simsimd_capability_t capabilities = simsimd_capabilities();

    int uses_neon = simsimd_uses_neon();
    int uses_sve = simsimd_uses_sve();
    int uses_haswell = simsimd_uses_haswell();
    int uses_skylake = simsimd_uses_skylake();
    int uses_ice = simsimd_uses_ice();
    int uses_sapphire = simsimd_uses_sapphire();

    assert(uses_neon == ((capabilities & simsimd_cap_neon_k) != 0));
    assert(uses_sve == ((capabilities & simsimd_cap_sve_k) != 0));
    assert(uses_haswell == ((capabilities & simsimd_cap_haswell_k) != 0));
    assert(uses_skylake == ((capabilities & simsimd_cap_skylake_k) != 0));
    assert(uses_ice == ((capabilities & simsimd_cap_ice_k) != 0));
    assert(uses_sapphire == ((capabilities & simsimd_cap_sapphire_k) != 0));
}

/**
 *  @brief  A trivial test that calls every implemented distance function and their dispatch versions
 *          on vectors A and B, where A and B are equal.
 */
void test_distance_from_itself() {
    simsimd_f64_t f64s[1536];
    simsimd_f32_t f32s[1536];
    simsimd_f16_t f16s[1536];
    simsimd_i8_t i8s[1536];
    simsimd_b8_t b8s[1536 / 8]; // 8 bits per word
    simsimd_distance_t distance;

    // Cosine distance between two vectors
    simsimd_cos_i8(i8s, i8s, 1536, &distance);
    simsimd_cos_f16(f16s, f16s, 1536, &distance);
    simsimd_cos_f32(f32s, f32s, 1536, &distance);
    simsimd_cos_f64(f64s, f64s, 1536, &distance);

    // Euclidean distance between two vectors
    simsimd_l2sq_i8(i8s, i8s, 1536, &distance);
    simsimd_l2sq_f16(f16s, f16s, 1536, &distance);
    simsimd_l2sq_f32(f32s, f32s, 1536, &distance);
    simsimd_l2sq_f64(f64s, f64s, 1536, &distance);

    // Inner product between two vectors
    simsimd_dot_f16(f16s, f16s, 1536, &distance);
    simsimd_dot_f32(f32s, f32s, 1536, &distance);
    simsimd_dot_f64(f64s, f64s, 1536, &distance);

    // Complex inner product between two vectors
    simsimd_dot_f16c(f16s, f16s, 1536, &distance);
    simsimd_dot_f32c(f32s, f32s, 1536, &distance);
    simsimd_dot_f64c(f64s, f64s, 1536, &distance);

    // Complex conjugate inner product between two vectors
    simsimd_vdot_f16c(f16s, f16s, 1536, &distance);
    simsimd_vdot_f32c(f32s, f32s, 1536, &distance);
    simsimd_vdot_f64c(f64s, f64s, 1536, &distance);

    // Hamming distance between two vectors
    simsimd_hamming_b8(b8s, b8s, 1536 / 8, &distance);

    // Jaccard distance between two vectors
    simsimd_jaccard_b8(b8s, b8s, 1536 / 8, &distance);

    // Jensen-Shannon divergence between two vectors
    simsimd_js_f16(f16s, f16s, 1536, &distance);
    simsimd_js_f32(f32s, f32s, 1536, &distance);
    simsimd_js_f64(f64s, f64s, 1536, &distance);

    // Kullback-Leibler divergence between two vectors
    simsimd_kl_f16(f16s, f16s, 1536, &distance);
    simsimd_kl_f32(f32s, f32s, 1536, &distance);
    simsimd_kl_f64(f64s, f64s, 1536, &distance);
}

int main(int argc, char** argv) {

    print_capabilities();
    test_utilities();
    test_distance_from_itself();
    return 0;
}