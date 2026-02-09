/**
 *  @brief Test suite entry point and configuration.
 *  @file test/test.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 */

#include <regex>
#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "numkong/capabilities.h" // nk_capabilities, nk_configure_thread

#include "test.hpp"
#include "test_cross.hpp"

static bool colors_enabled() {
    static bool const result = [] {
        if (std::getenv("NO_COLOR")) return false;
        if (std::getenv("FORCE_COLOR")) return true;
#if !defined(_WIN32)
        return isatty(fileno(stdout)) != 0;
#else
        return false;
#endif
    }();
    return result;
}

static void print_indicator(bool on) {
    if (on) std::printf(colors_enabled() ? "\033[32m\xe2\x97\x8f\033[0m" : "\xe2\x97\x8f");
    else std::printf(colors_enabled() ? "\033[2m\xe2\x97\x8b\033[0m" : "\xe2\x97\x8b");
}

static void print_isa(char const *name, int compiled, nk_capability_t cap, nk_capability_t runtime) {
    if (!compiled) return;
    std::printf("  %s ", name);
    print_indicator((runtime & cap) != 0);
}

// Definition of global variables declared extern in test.hpp
std::size_t dense_dimensions = 1024; // For dot products, spatial metrics
std::size_t sparse_dimensions = 256; // For sparse set intersection and sparse dot
std::size_t mesh_points = 256;       // For RMSD, Kabsch (3D point clouds)
std::size_t matrix_height = 64, matrix_width = 64, matrix_depth = 64;

test_config_t global_config;
std::size_t global_failure_count = 0;

bool test_config_t::should_run(char const *test_name) const {
    if (!filter) return true;
    try {
        std::regex pattern(filter);
        return std::regex_search(test_name, pattern);
    }
    catch (std::regex_error const &) {
        return std::strstr(test_name, filter) != nullptr;
    }
}

/**
 *  @brief Print the header for the error statistics table.
 */
void print_stats_header() noexcept {
    std::puts("");
    std::printf("NumKong Precision Analysis\n");
    std::printf("%-40s %12s %10s %12s %12s %10s\n", "Kernel", "max_ulp", "mean_ulp", "max_abs", "max_rel", "exact");
    std::printf("\n");
}

/**
 *  @brief Test FP8 (e4m3) conversions round-trip accuracy.
 */
void test_fp8_conversions() {
    float test_values[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 0.125f, 8.0f};
    error_stats_t stats;
    for (float expected : test_values) {
        nk_e4m3_t e4m3;
        float roundtrip;
        nk_f32_to_e4m3(&expected, &e4m3);
        nk_e4m3_to_f32(&e4m3, &roundtrip);
        stats.accumulate(f32_t(roundtrip), f32_t(expected));
    }
    stats.report("fp8_e4m3_roundtrip");
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    if (char const *env = std::getenv("NK_TEST_ASSERT")) global_config.assert_on_failure = std::atoi(env) != 0;
    if (char const *env = std::getenv("NK_TEST_VERBOSE")) global_config.verbose = std::atoi(env) != 0;
    if (char const *env = std::getenv("NK_TEST_ULP_THRESHOLD_F32")) global_config.ulp_threshold_f32 = std::atoll(env);
    if (char const *env = std::getenv("NK_TEST_ULP_THRESHOLD_F16")) global_config.ulp_threshold_f16 = std::atoll(env);
    if (char const *env = std::getenv("NK_TEST_ULP_THRESHOLD_BF16")) global_config.ulp_threshold_bf16 = std::atoll(env);
    if (char const *env = std::getenv("NK_TEST_TIME_BUDGET_MS")) global_config.time_budget_ms = std::atoll(env);
    if (char const *env = std::getenv("NK_SEED")) global_config.seed = std::atoll(env);
    global_config.filter = std::getenv("NK_FILTER"); // e.g., "dot", "angular", "kld"
    if (char const *env = std::getenv("NK_TEST_DISTRIBUTION")) {
        if (std::strcmp(env, "uniform_k") == 0) global_config.distribution = random_distribution_kind_t::uniform_k;
        else if (std::strcmp(env, "cauchy_k") == 0) global_config.distribution = random_distribution_kind_t::cauchy_k;
        else if (std::strcmp(env, "lognormal_k") == 0)
            global_config.distribution = random_distribution_kind_t::lognormal_k;
    }

    // Parse dimension overrides from environment variables
    if (char const *env = std::getenv("NK_DENSE_DIMENSIONS")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) dense_dimensions = val;
    }
    if (char const *env = std::getenv("NK_SPARSE_DIMENSIONS")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) sparse_dimensions = val;
    }
    if (char const *env = std::getenv("NK_MESH_POINTS")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) mesh_points = val;
    }
    if (char const *env = std::getenv("NK_MATRIX_HEIGHT")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) matrix_height = val;
    }
    if (char const *env = std::getenv("NK_MATRIX_WIDTH")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) matrix_width = val;
    }
    if (char const *env = std::getenv("NK_MATRIX_DEPTH")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) matrix_depth = val;
    }

    nk_capability_t runtime_caps = nk_capabilities();
    nk_configure_thread(runtime_caps); // Also enables AMX if available

    std::printf(colors_enabled() ? "\033[1mNumKong Precision Testing Suite v%d.%d.%d\033[0m\n"
                                 : "NumKong Precision Testing Suite v%d.%d.%d\n",
                NK_VERSION_MAJOR, NK_VERSION_MINOR, NK_VERSION_PATCH);

    // Compilation row
    std::printf("  Compilation: F16 ");
    print_indicator(NK_NATIVE_F16);
    std::printf("  BF16 ");
    print_indicator(NK_NATIVE_BF16);
    std::printf("  MKL ");
    print_indicator(NK_COMPARE_TO_MKL);
    std::printf("  OpenMP ");
#if NK_TEST_USE_OPENMP
    print_indicator(true);
#else
    print_indicator(false);
#endif
    std::printf("\n");

    // ISA row — one print_isa() call per ISA, skips those not compiled
    std::printf("  ISA:");
    // x86
    print_isa("Haswell", NK_TARGET_HASWELL, nk_cap_haswell_k, runtime_caps);
    print_isa("Skylake", NK_TARGET_SKYLAKE, nk_cap_skylake_k, runtime_caps);
    print_isa("Ice Lake", NK_TARGET_ICELAKE, nk_cap_icelake_k, runtime_caps);
    print_isa("Genoa", NK_TARGET_GENOA, nk_cap_genoa_k, runtime_caps);
    print_isa("Sapphire", NK_TARGET_SAPPHIRE, nk_cap_sapphire_k, runtime_caps);
    print_isa("Sapphire AMX", NK_TARGET_SAPPHIREAMX, nk_cap_sapphireamx_k, runtime_caps);
    print_isa("Granite AMX", NK_TARGET_GRANITEAMX, nk_cap_graniteamx_k, runtime_caps);
    print_isa("Turin", NK_TARGET_TURIN, nk_cap_turin_k, runtime_caps);
    print_isa("Sierra", NK_TARGET_SIERRA, nk_cap_sierra_k, runtime_caps);
    // Arm
    print_isa("NEON", NK_TARGET_NEON, nk_cap_neon_k, runtime_caps);
    print_isa("NEON F16", NK_TARGET_NEONHALF, nk_cap_neonhalf_k, runtime_caps);
    print_isa("NEON BF16", NK_TARGET_NEONBFDOT, nk_cap_neonbfdot_k, runtime_caps);
    print_isa("NEON I8", NK_TARGET_NEONSDOT, nk_cap_neonsdot_k, runtime_caps);
    print_isa("NEON FHM", NK_TARGET_NEONFHM, nk_cap_neonfhm_k, runtime_caps);
    print_isa("SVE", NK_TARGET_SVE, nk_cap_sve_k, runtime_caps);
    print_isa("SVE F16", NK_TARGET_SVEHALF, nk_cap_svehalf_k, runtime_caps);
    print_isa("SVE BF16", NK_TARGET_SVEBFDOT, nk_cap_svebfdot_k, runtime_caps);
    print_isa("SVE I8", NK_TARGET_SVESDOT, nk_cap_svesdot_k, runtime_caps);
    print_isa("SVE2", NK_TARGET_SVE2, nk_cap_sve2_k, runtime_caps);
    print_isa("SVE2P1", NK_TARGET_SVE2P1, nk_cap_sve2p1_k, runtime_caps);
    print_isa("SME", NK_TARGET_SME, nk_cap_sme_k, runtime_caps);
    print_isa("SME2", NK_TARGET_SME2, nk_cap_sme2_k, runtime_caps);
    print_isa("SME2P1", NK_TARGET_SME2P1, nk_cap_sme2p1_k, runtime_caps);
    print_isa("SME F64", NK_TARGET_SMEF64, nk_cap_smef64_k, runtime_caps);
    print_isa("SME F16", NK_TARGET_SMEHALF, nk_cap_smehalf_k, runtime_caps);
    print_isa("SME BF16", NK_TARGET_SMEBF16, nk_cap_smebf16_k, runtime_caps);
    print_isa("SME FA64", NK_TARGET_SMEFA64, nk_cap_smefa64_k, runtime_caps);
    print_isa("SME LUT2", NK_TARGET_SMELUT2, nk_cap_smelut2_k, runtime_caps);
    // RISC-V
    print_isa("RVV", NK_TARGET_RVV, nk_cap_rvv_k, runtime_caps);
    print_isa("RVV HALF", NK_TARGET_RVVHALF, nk_cap_rvvhalf_k, runtime_caps);
    print_isa("RVV BF16", NK_TARGET_RVVBF16, nk_cap_rvvbf16_k, runtime_caps);
    // WASM
    print_isa("V128 Relaxed", NK_TARGET_V128RELAXED, nk_cap_v128relaxed_k, runtime_caps);
    std::printf("\n");

    // Dimensions row
    std::printf("  Dimensions: dense=%zu  sparse=%zu  mesh=%zu  matrix=%zux%zux%zu\n", dense_dimensions,
                sparse_dimensions, mesh_points, matrix_height, matrix_width, matrix_depth);

    // ULP thresholds
    std::printf("  ULP: f32 \xe2\x89\xa4 %llu  f16 \xe2\x89\xa4 %llu  bf16 \xe2\x89\xa4 %llu\n",
                (unsigned long long)global_config.ulp_threshold_f32,
                (unsigned long long)global_config.ulp_threshold_f16,
                (unsigned long long)global_config.ulp_threshold_bf16);

    // Test-specific config
    std::printf("  Test: seed=%u  budget=%zums  distribution=%s  assert=%s\n", global_config.seed,
                global_config.time_budget_ms, global_config.distribution_name(),
                global_config.assert_on_failure ? "on" : "off");
    std::printf("\n");

    test_vector_types();

    // Print a table header
    print_stats_header();
    test_fp8_conversions();
    test_casts();

    // Core operation tests
    test_dot();
    test_spatial();
    test_curved();
    test_probability();
    test_set();
    test_elementwise();
    test_trigonometry();

    // Additional operation tests
    test_reduce();
    test_geospatial();
    test_mesh();
    test_sparse();

    // Cross/batch tests (ISA-family files for parallel compilation)
    test_cross_serial();
    test_cross_x86();
    test_cross_amx();
    test_cross_arm();
    test_cross_sme();
    test_cross_blas();

    if (global_failure_count > 0) {
        std::puts("");
        std::printf("%zu kernel(s) exceeded ULP thresholds.\n", global_failure_count);
        return 1;
    }
    std::puts("");
    std::printf("All tests passed.\n");
    return 0;
}
