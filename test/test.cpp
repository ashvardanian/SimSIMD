/**
 *  @brief Test suite entry point and configuration.
 *  @file test/test.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 */

#if defined(_WIN32)
#include <regex>
#else
#include <regex.h>
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

test_config_t global_config;

bool test_config_t::should_run(char const *test_name) const {
    if (!filter) return true;
#if defined(_WIN32)
    try {
        std::regex pattern(filter);
        return std::regex_search(test_name, pattern);
    }
    catch (std::regex_error const &) {
        return std::strstr(test_name, filter) != nullptr;
    }
#else
    regex_t pattern;
    int rc = regcomp(&pattern, filter, REG_EXTENDED | REG_NOSUB);
    if (rc != 0) return std::strstr(test_name, filter) != nullptr;
    rc = regexec(&pattern, test_name, 0, nullptr, 0);
    regfree(&pattern);
    return rc == 0;
#endif
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

    // Parse CLI arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--filter=", 9) == 0) { global_config.filter = argv[i] + 9; }
        else if (std::strcmp(argv[i], "--filter") == 0 && i + 1 < argc) { global_config.filter = argv[++i]; }
        else if (std::strcmp(argv[i], "--assert") == 0) { global_config.assert_on_failure = true; }
        else if (std::strcmp(argv[i], "--verbose") == 0) { global_config.verbose = true; }
        else if (std::strncmp(argv[i], "--time-budget=", 14) == 0) {
            global_config.time_budget_ms = static_cast<std::size_t>(std::atoll(argv[i] + 14));
        }
        else if (std::strcmp(argv[i], "--time-budget") == 0 && i + 1 < argc) {
            global_config.time_budget_ms = static_cast<std::size_t>(std::atoll(argv[++i]));
        }
        // Foreign flags from GTest
        else if (std::strncmp(argv[i], "--gtest_filter=", 15) == 0) {
            global_config.filter = argv[i] + 15;
            std::fprintf(stderr, "Note: Mapped --gtest_filter to --filter. Prefer: --filter='%s'\n",
                         global_config.filter);
        }
        else if (std::strncmp(argv[i], "--gtest_", 8) == 0) {
            std::fprintf(stderr, "Note: GTest flag '%s' is not supported in nk_test. Ignoring.\n", argv[i]);
        }
        // Foreign flags from Google Benchmark
        else if (std::strncmp(argv[i], "--benchmark_filter=", 19) == 0) {
            global_config.filter = argv[i] + 19;
            std::fprintf(stderr, "Note: Mapped --benchmark_filter to --filter. Prefer: --filter='%s'\n",
                         global_config.filter);
        }
        else if (std::strncmp(argv[i], "--benchmark_min_time=", 21) == 0) {
            // Parse value, stripping trailing 's' if present (e.g., "10s" -> 10000 ms)
            char const *val = argv[i] + 21;
            double seconds = std::atof(val);
            global_config.time_budget_ms = static_cast<std::size_t>(seconds * 1000);
            std::fprintf(stderr, "Note: Mapped --benchmark_min_time to --time-budget. Prefer: --time-budget=%zu\n",
                         global_config.time_budget_ms);
        }
        else if (std::strncmp(argv[i], "--benchmark_", 12) == 0) {
            std::fprintf(stderr, "Note: Google Benchmark flag '%s' is not supported in nk_test. Ignoring.\n", argv[i]);
        }
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::fprintf( //
                stdout,
                "Usage: nk_test [--filter=<regex>] [--time-budget=<ms>] [--assert] [--verbose] [--help]\n" //
                "\n"                                                                                       //
                "Arguments:\n"                                                                             //
                "  --filter=<regex>     Filter tests by name (regex or substring)\n"                       //
                "  --time-budget=<ms>   Time budget per kernel in milliseconds (default: 1000)\n"          //
                "  --assert             Abort on first failure\n"                                          //
                "  --verbose            Verbose output\n"                                                  //
                "\n"                                                                                       //
                "Environment Variables:\n"                                                                 //
                "  NK_FILTER=<regex>          Same as --filter\n"                                          //
                "  NK_BUDGET_SECS=<seconds>   Time budget per kernel (default: 1)\n"                       //
                "  NK_SEED=<int>              Random seed\n"                                               //
                "  NK_IN_QEMU=1               Skip unreliable half-precision tests\n"                      //
                "  NK_TEST_ASSERT=1           Same as --assert\n"                                          //
                "  NK_TEST_VERBOSE=1          Same as --verbose\n"                                         //
                "  NK_ULP_THRESHOLD_F32=N     ULP tolerance for f32\n"                                     //
                "  NK_ULP_THRESHOLD_F16=N     ULP tolerance for f16\n"                                     //
                "  NK_ULP_THRESHOLD_BF16=N    ULP tolerance for bf16\n"                                    //
                "  NK_RANDOM_DISTRIBUTION=X   uniform_k, cauchy_k, lognormal_k\n"                          //
                "  NK_DENSE_DIMENSIONS=N      Override dense vector dimensions\n"                          //
                "  NK_CURVED_DIMENSIONS=N     Override curved vector dimensions\n"                         //
                "  NK_SPARSE_DIMENSIONS=N     Override sparse vector dimensions\n");                       //
            return 0;
        }
        else {
            std::fprintf(stderr, "Error: unrecognized argument '%s'. Try --help.\n", argv[i]);
            return 1;
        }
    }

    if (std::getenv("NK_IN_QEMU")) global_config.running_in_qemu = true;
    if (char const *env = std::getenv("NK_TEST_ASSERT")) global_config.assert_on_failure = std::atoi(env) != 0;
    if (char const *env = std::getenv("NK_TEST_VERBOSE")) global_config.verbose = std::atoi(env) != 0;
    if (char const *env = std::getenv("NK_ULP_THRESHOLD_F32")) global_config.ulp_threshold_f32 = std::atoll(env);
    if (char const *env = std::getenv("NK_ULP_THRESHOLD_F16")) global_config.ulp_threshold_f16 = std::atoll(env);
    if (char const *env = std::getenv("NK_ULP_THRESHOLD_BF16")) global_config.ulp_threshold_bf16 = std::atoll(env);
    if (char const *env = std::getenv("NK_SEED")) global_config.seed = std::atoll(env);
    if (!global_config.filter) global_config.filter = std::getenv("NK_FILTER"); // e.g., "dot", "angular", "kld"

    if (global_config.time_budget_ms == 1000) {
        if (char const *env = std::getenv("NK_BUDGET_SECS")) {
            double seconds = std::atof(env);
            if (seconds > 0) global_config.time_budget_ms = static_cast<std::size_t>(seconds * 1000);
        }
    }

    if (char const *env = std::getenv("NK_RANDOM_DISTRIBUTION")) {
        if (std::strcmp(env, "uniform_k") == 0) global_config.distribution = random_distribution_kind_t::uniform_k;
        else if (std::strcmp(env, "cauchy_k") == 0) global_config.distribution = random_distribution_kind_t::cauchy_k;
        else if (std::strcmp(env, "lognormal_k") == 0)
            global_config.distribution = random_distribution_kind_t::lognormal_k;
    }

    // Parse dimension overrides from environment variables
    if (char const *env = std::getenv("NK_DENSE_DIMENSIONS")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) global_config.dense_dimensions = val;
    }
    if (char const *env = std::getenv("NK_CURVED_DIMENSIONS")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) global_config.curved_dimensions = val;
    }
    if (char const *env = std::getenv("NK_SPARSE_DIMENSIONS")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) global_config.sparse_dimensions = val;
    }
    if (char const *env = std::getenv("NK_MESH_POINTS")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) global_config.mesh_points = val;
    }
    if (char const *env = std::getenv("NK_MATRIX_HEIGHT")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) global_config.matrix_height = val;
    }
    if (char const *env = std::getenv("NK_MATRIX_WIDTH")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) global_config.matrix_width = val;
    }
    if (char const *env = std::getenv("NK_MATRIX_DEPTH")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) global_config.matrix_depth = val;
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
    print_isa("RVV BB", NK_TARGET_RVVBB, nk_cap_rvvbb_k, runtime_caps);
    // WASM
    print_isa("V128 Relaxed", NK_TARGET_V128RELAXED, nk_cap_v128relaxed_k, runtime_caps);
    std::printf("\n");

    // Dimensions row
    std::printf("  Dimensions: dense=%zu  curved=%zu  sparse=%zu  mesh=%zu  matrix=%zux%zux%zu\n",
                global_config.dense_dimensions, global_config.curved_dimensions, global_config.sparse_dimensions,
                global_config.mesh_points, global_config.matrix_height, global_config.matrix_width,
                global_config.matrix_depth);

    // ULP thresholds
    std::printf("  ULP: f32 \xe2\x89\xa4 %llu  f16 \xe2\x89\xa4 %llu  bf16 \xe2\x89\xa4 %llu\n",
                (unsigned long long)global_config.ulp_threshold_f32,
                (unsigned long long)global_config.ulp_threshold_f16,
                (unsigned long long)global_config.ulp_threshold_bf16);

    // Test-specific config
    std::printf("  Test: seed=%u  budget=%zums  distribution=%s  assert=%s  qemu=%s\n", global_config.seed,
                global_config.time_budget_ms, global_config.distribution_name(),
                global_config.assert_on_failure ? "on" : "off", global_config.running_in_qemu ? "yes" : "no");
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
    test_cross_rvv();

    if (global_config.failure_count > 0) {
        std::puts("");
        std::printf("%zu kernel(s) exceeded ULP thresholds.\n", global_config.failure_count);
        return 1;
    }
    std::puts("");
    std::printf("All tests passed.\n");
    return 0;
}
