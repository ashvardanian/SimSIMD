/**
 *  @brief NumKong C++ Benchmark Suite using Google Benchmark - Main entry point.
 *  @file bench/bench.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *
 *  Comprehensive benchmarks for NumKong SIMD-optimized functions measuring
 *  throughput performance. Run with:
 *
 *  ```bash
 *  cmake -B build_release -D NK_BUILD_BENCH=1
 *  cmake --build build_release
 *  build_release/nk_bench
 *  ```
 *
 *  Environment Variables:
 *    NK_FILTER=<pattern>           - Filter benchmarks by name regex (default: run all)
 *    NK_SEED=N                     - RNG seed (default: 42)
 *
 *    NK_DENSE_DIMENSIONS=N         - Vector dimension for dot/spatial benchmarks (default: 1536)
 *    NK_MESH_POINTS=N              - Point count for mesh benchmarks (default: 1000)
 *    NK_MATRIX_HEIGHT=N            - GEMM M dimension (default: 1024), like dataset size for kNN
 *    NK_MATRIX_WIDTH=N             - GEMM N dimension (default: 128), like queries count for kNN
 *    NK_MATRIX_DEPTH=N             - GEMM K dimension (default: 1536), like vector dimensions in KNN
 *
 *    NK_CURVED_DIMENSIONS=N        - Vector dimension for curved benchmarks (default: 64)
 *    NK_SPARSE_FIRST_LENGTH=N      - First set size for sparse benchmarks (default: 1024)
 *    NK_SPARSE_SECOND_LENGTH=N     - Second set size for sparse benchmarks (default: 8192)
 *    NK_SPARSE_INTERSECTION=F      - Intersection share 0.0-1.0 (default: 0.5)
 */

#include <cstdio>  // std::printf
#include <cstdlib> // std::getenv, std::atoll, std::atof
#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "numkong/capabilities.h" // Runtime capability detection

#include "bench.hpp"

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

// Global variable definitions
std::size_t dense_dimensions = 1536;
std::size_t curved_dimensions = 64;
std::size_t mesh_points = 1000;
std::size_t matrix_height = 1024, matrix_width = 128, matrix_depth = 1536;
std::uint32_t random_seed = 42;
std::size_t sparse_first_length = 1024;
std::size_t sparse_second_length = 8192;
double sparse_intersection_share = 0.5;

int main(int argc, char **argv) {
    nk_capability_t runtime_caps = nk_capabilities();
    nk_configure_thread(runtime_caps); // Also enables AMX if available

#if NK_COMPARE_TO_MKL
    mkl_set_num_threads(1);
#elif NK_COMPARE_TO_BLAS
    if (openblas_set_num_threads) openblas_set_num_threads(1);
#endif

    // Override dimensions from environment variables if provided
    if (char const *env_dense = std::getenv("NK_DENSE_DIMENSIONS")) {
        std::size_t parsed_dense = static_cast<std::size_t>(std::atoll(env_dense));
        if (parsed_dense > 0) dense_dimensions = parsed_dense;
    }
    if (char const *env_curved = std::getenv("NK_CURVED_DIMENSIONS")) {
        std::size_t parsed_curved = static_cast<std::size_t>(std::atoll(env_curved));
        if (parsed_curved > 0) curved_dimensions = parsed_curved;
    }
    if (char const *env_mesh = std::getenv("NK_MESH_POINTS")) {
        std::size_t parsed_mesh = static_cast<std::size_t>(std::atoll(env_mesh));
        if (parsed_mesh > 0) mesh_points = parsed_mesh;
    }
    if (char const *env_matrix_height = std::getenv("NK_MATRIX_HEIGHT")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_matrix_height));
        if (parsed > 0) matrix_height = parsed;
    }
    if (char const *env_matrix_width = std::getenv("NK_MATRIX_WIDTH")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_matrix_width));
        if (parsed > 0) matrix_width = parsed;
    }
    if (char const *env_matrix_depth = std::getenv("NK_MATRIX_DEPTH")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_matrix_depth));
        if (parsed > 0) matrix_depth = parsed;
    }
    if (char const *env_seed = std::getenv("NK_SEED")) {
        std::uint32_t parsed = static_cast<std::uint32_t>(std::atoll(env_seed));
        random_seed = parsed;
    }
    if (char const *env_sparse_first = std::getenv("NK_SPARSE_FIRST_LENGTH")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_sparse_first));
        if (parsed > 0) sparse_first_length = parsed;
    }
    if (char const *env_sparse_second = std::getenv("NK_SPARSE_SECOND_LENGTH")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_sparse_second));
        if (parsed > 0) sparse_second_length = parsed;
    }
    if (char const *env_sparse_intersection = std::getenv("NK_SPARSE_INTERSECTION")) {
        double parsed = std::atof(env_sparse_intersection);
        if (parsed >= 0.0 && parsed <= 1.0) sparse_intersection_share = parsed;
    }

    std::printf(colors_enabled() ? "\033[1mNumKong Benchmarking Suite v%d.%d.%d\033[0m\n"
                                 : "NumKong Benchmarking Suite v%d.%d.%d\n",
                NK_VERSION_MAJOR, NK_VERSION_MINOR, NK_VERSION_PATCH);

    // Compilation row
    std::printf("  Compilation: F16 ");
    print_indicator(NK_NATIVE_F16);
    std::printf("  BF16 ");
    print_indicator(NK_NATIVE_BF16);
    std::printf("  MKL ");
    print_indicator(NK_COMPARE_TO_MKL);
    std::printf("  OpenMP ");
#ifdef _OPENMP
    print_indicator(true);
#else
    print_indicator(false);
#endif
    std::printf("\n");

    // ISA row
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
    std::printf("  Dimensions: dense=%zu  curved=%zu  mesh=%zu  matrix=%zux%zux%zu  sparse=%zu/%zu@%.2f\n",
                dense_dimensions, curved_dimensions, mesh_points, matrix_height, matrix_width, matrix_depth,
                sparse_first_length, sparse_second_length, sparse_intersection_share);

    // Bench-specific config
    std::printf("  Bench: seed=%u\n", random_seed);
    std::printf("\n");

    // Handle NK_FILTER environment variable by injecting --benchmark_filter argument
    std::vector<char *> modified_argv(argv, argv + argc);
    std::string filter_arg;
    if (char const *env_filter = std::getenv("NK_FILTER")) {
        filter_arg = std::string("--benchmark_filter=") + env_filter;
        modified_argv.push_back(const_cast<char *>(filter_arg.c_str()));
        std::printf("Applying benchmark filter from NK_FILTER: %s\n\n", env_filter);
    }
    int modified_argc = static_cast<int>(modified_argv.size());
    char **modified_argv_ptr = modified_argv.data();

    // Run the benchmarks
    bm::Initialize(&modified_argc, modified_argv_ptr);
    if (bm::ReportUnrecognizedArguments(modified_argc, modified_argv_ptr)) return 1;

    // Register all benchmarks from split files
    bench_dot();
    bench_spatial();
    bench_set();
    bench_curved();
    bench_probability();
    bench_each();
    bench_trigonometry();
    bench_geospatial();
    bench_mesh();
    bench_sparse();
    bench_cast();
    bench_reduce();

    // Register cross/batch benchmarks (ISA-family files for parallel compilation)
    bench_cross_serial();
    bench_cross_x86();
    bench_cross_amx();
    bench_cross_arm();
    bench_cross_sme();
    bench_cross_blas();

    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}
