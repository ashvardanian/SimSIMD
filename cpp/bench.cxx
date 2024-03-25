#include <cmath>  // `std::sqrt`
#include <thread> // `std::thread`

#include <benchmark/benchmark.h>

#if !defined(SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS)
#define SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS 0
#endif
#if SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS
#include <cblas.h>
#endif

#define SIMSIMD_NATIVE_F16 0
#define SIMSIMD_RSQRT(x) (1 / sqrtf(x))
#define SIMSIMD_LOG(x) (logf(x))
#include <simsimd/simsimd.h>

namespace bm = benchmark;

template <typename scalar_at, std::size_t dimensions_ak> struct vectors_pair_gt {
    scalar_at a[dimensions_ak]{};
    scalar_at b[dimensions_ak]{};

    std::size_t dimensions() const noexcept { return dimensions_ak; }
    std::size_t size_bytes() const noexcept { return dimensions_ak * sizeof(scalar_at); }

    void set(scalar_at v) noexcept {
        for (std::size_t i = 0; i != dimensions_ak; ++i)
            a[i] = b[i] = v;
    }

    void randomize() noexcept {

        double a2_sum = 0, b2_sum = 0;
        for (std::size_t i = 0; i != dimensions_ak; ++i) {
            if constexpr (std::is_integral_v<scalar_at>)
                a[i] = static_cast<scalar_at>(rand()), b[i] = static_cast<scalar_at>(rand());
            else {
                double ai = double(rand()) / double(RAND_MAX), bi = double(rand()) / double(RAND_MAX);
                a2_sum += ai * ai, b2_sum += bi * bi;
                a[i] = static_cast<scalar_at>(ai), b[i] = static_cast<scalar_at>(bi);
            }
        }

        // Normalize the vectors:
        if constexpr (!std::is_integral_v<scalar_at>) {
            a2_sum = std::sqrt(a2_sum);
            b2_sum = std::sqrt(b2_sum);
            for (std::size_t i = 0; i != dimensions_ak; ++i)
                a[i] = static_cast<scalar_at>(a[i] / a2_sum), b[i] = static_cast<scalar_at>(b[i] / b2_sum);
        }
    }
};

template <typename pair_at, typename metric_at = void>
constexpr void measure(bm::State& state, metric_at metric, metric_at baseline) {

    pair_at pair;
    pair.randomize();
    // pair.set(1);

    auto call_baseline = [&](pair_at& pair) -> simsimd_f32_t {
        // Output for real vectors have a single dimensions.
        // Output for complex vectors have two dimensions.
        simsimd_distance_t results[2] = {0, 0};
        baseline(pair.a, pair.b, pair.dimensions(), &results[0]);
        return results[0] + results[1];
    };
    auto call_contender = [&](pair_at& pair) -> simsimd_f32_t {
        // Output for real vectors have a single dimensions.
        // Output for complex vectors have two dimensions.
        simsimd_distance_t results[2] = {0, 0};
        metric(pair.a, pair.b, pair.dimensions(), &results[0]);
        return results[0] + results[1];
    };

    double c_baseline = call_baseline(pair);
    double c = 0;
    std::size_t iterations = 0;
    for (auto _ : state)
        bm::DoNotOptimize((c = call_contender(pair))), iterations++;

    state.counters["bytes"] = bm::Counter(iterations * pair.size_bytes() * 2, bm::Counter::kIsRate);
    state.counters["pairs"] = bm::Counter(iterations, bm::Counter::kIsRate);

    double delta = std::abs(c - c_baseline) > 0.0001 ? std::abs(c - c_baseline) : 0;
    double error = delta != 0 && c_baseline != 0 ? delta / c_baseline : 0;
    state.counters["abs_delta"] = delta;
    state.counters["relative_error"] = error;
}

template <typename scalar_at, typename metric_at = void>
void register_(std::string name, metric_at* distance_func, metric_at* baseline_func) {

    std::size_t seconds = 10;
    std::size_t threads = std::thread::hardware_concurrency(); // 1;

    using pair_dims_t = vectors_pair_gt<scalar_at, 1536>;
    std::string name_dims = name + "_" + std::to_string(pair_dims_t{}.dimensions()) + "d";
    bm::RegisterBenchmark(name_dims.c_str(), measure<pair_dims_t, metric_at*>, distance_func, baseline_func)
        ->MinTime(seconds)
        ->Threads(threads);

    return;
    using pair_bytes_t = vectors_pair_gt<scalar_at, 1536 / sizeof(scalar_at)>;
    std::string name_bytes = name + "_" + std::to_string(pair_bytes_t{}.size_bytes()) + "b";
    bm::RegisterBenchmark(name_bytes.c_str(), measure<pair_bytes_t, metric_at*>, distance_func, baseline_func)
        ->MinTime(seconds)
        ->Threads(threads);
}

#if SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS

void dot_f32_blas(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    *result = cblas_sdot((int)n, a, 1, b, 1);
}

void dot_f64_blas(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    *result = cblas_ddot((int)n, a, 1, b, 1);
}

void dot_f32c_blas(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    simsimd_f32_t f32_result[2] = {0, 0};
    cblas_cdotu_sub((int)n / 2, a, 1, b, 1, f32_result);
    result[0] = f32_result[0];
    result[1] = f32_result[1];
}

void dot_f64c_blas(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    cblas_zdotu_sub((int)n / 2, a, 1, b, 1, result);
}

void vdot_f32c_blas(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    simsimd_f32_t f32_result[2] = {0, 0};
    cblas_cdotc_sub((int)n / 2, a, 1, b, 1, f32_result);
    result[0] = f32_result[0];
    result[1] = f32_result[1];
}

void vdot_f64c_blas(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    cblas_zdotc_sub((int)n / 2, a, 1, b, 1, result);
}

#endif

int main(int argc, char** argv) {
    simsimd_capability_t runtime_caps = simsimd_capabilities();

    // Log supported functionality
    char const* flags[2] = {"false", "true"};
    std::printf("Benchmarking Similarity Measures\n");
    std::printf("- Compiler used native F16: %s\n", flags[SIMSIMD_NATIVE_F16]);
    std::printf("- Benchmark against CBLAS: %s\n", flags[SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS]);
    std::printf("\n");
    std::printf("Compile-time settings:\n");
    std::printf("- Arm NEON support enabled: %s\n", flags[SIMSIMD_TARGET_NEON]);
    std::printf("- Arm SVE support enabled: %s\n", flags[SIMSIMD_TARGET_SVE]);
    std::printf("- x86 Haswell support enabled: %s\n", flags[SIMSIMD_TARGET_HASWELL]);
    std::printf("- x86 Skylake support enabled: %s\n", flags[SIMSIMD_TARGET_SKYLAKE]);
    std::printf("- x86 Ice Lake support enabled: %s\n", flags[SIMSIMD_TARGET_ICE]);
    std::printf("- x86 Sapphire Rapids support enabled: %s\n", flags[SIMSIMD_TARGET_SAPPHIRE]);
    std::printf("\n");
    std::printf("Run-time settings:\n");
    std::printf("- Arm NEON support enabled: %s\n", flags[(runtime_caps & simsimd_cap_neon_k) != 0]);
    std::printf("- Arm SVE support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sve_k) != 0]);
    std::printf("- x86 Haswell support enabled: %s\n", flags[(runtime_caps & simsimd_cap_haswell_k) != 0]);
    std::printf("- x86 Skylake support enabled: %s\n", flags[(runtime_caps & simsimd_cap_skylake_k) != 0]);
    std::printf("- x86 Ice Lake support enabled: %s\n", flags[(runtime_caps & simsimd_cap_ice_k) != 0]);
    std::printf("- x86 Sapphire Rapids support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sapphire_k) != 0]);
    std::printf("\n");

    // Run the benchmarks
    bm::Initialize(&argc, argv);
    if (bm::ReportUnrecognizedArguments(argc, argv))
        return 1;

#if SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS

    register_<simsimd_f32_t>("dot_f32_blas", dot_f32_blas, simsimd_dot_f32_accurate);
    register_<simsimd_f64_t>("dot_f64_blas", dot_f64_blas, simsimd_dot_f64_serial);
    register_<simsimd_f32_t>("dot_f32c_blas", dot_f32c_blas, simsimd_dot_f32c_accurate);
    register_<simsimd_f64_t>("dot_f64c_blas", dot_f64c_blas, simsimd_dot_f64c_serial);
    register_<simsimd_f32_t>("vdot_f32c_blas", vdot_f32c_blas, simsimd_vdot_f32c_accurate);
    register_<simsimd_f64_t>("vdot_f64c_blas", vdot_f64c_blas, simsimd_vdot_f64c_serial);

#endif

#if SIMSIMD_TARGET_NEON

    register_<simsimd_f16_t>("dot_f16_neon", simsimd_dot_f16_neon, simsimd_dot_f16_accurate);
    register_<simsimd_f16_t>("cos_f16_neon", simsimd_cos_f16_neon, simsimd_cos_f16_accurate);
    register_<simsimd_f16_t>("l2sq_f16_neon", simsimd_l2sq_f16_neon, simsimd_l2sq_f16_accurate);
    register_<simsimd_f16_t>("kl_f16_neon", simsimd_kl_f16_neon, simsimd_kl_f16_accurate);
    register_<simsimd_f16_t>("js_f16_neon", simsimd_js_f16_neon, simsimd_js_f16_accurate);

    register_<simsimd_f32_t>("dot_f32_neon", simsimd_dot_f32_neon, simsimd_dot_f32_accurate);
    register_<simsimd_f32_t>("cos_f32_neon", simsimd_cos_f32_neon, simsimd_cos_f32_accurate);
    register_<simsimd_f32_t>("l2sq_f32_neon", simsimd_l2sq_f32_neon, simsimd_l2sq_f32_accurate);
    register_<simsimd_f32_t>("kl_f32_neon", simsimd_kl_f32_neon, simsimd_kl_f32_accurate);
    register_<simsimd_f32_t>("js_f32_neon", simsimd_js_f32_neon, simsimd_js_f32_accurate);

    register_<simsimd_i8_t>("cos_i8_neon", simsimd_cos_i8_neon, simsimd_cos_i8_accurate);
    register_<simsimd_i8_t>("l2sq_i8_neon", simsimd_l2sq_i8_neon, simsimd_l2sq_i8_accurate);

    register_<simsimd_b8_t>("hamming_b8_neon", simsimd_hamming_b8_neon, simsimd_hamming_b8_serial);
    register_<simsimd_b8_t>("jaccard_b8_neon", simsimd_jaccard_b8_neon, simsimd_jaccard_b8_serial);

    register_<simsimd_f16_t>("dot_f16c_neon", simsimd_dot_f16c_neon, simsimd_dot_f16c_accurate);
    register_<simsimd_f16_t>("vdot_f16c_neon", simsimd_vdot_f16c_neon, simsimd_vdot_f16c_accurate);
    register_<simsimd_f32_t>("dot_f32c_neon", simsimd_dot_f32c_neon, simsimd_dot_f32c_accurate);
    register_<simsimd_f32_t>("vdot_f32c_neon", simsimd_vdot_f32c_neon, simsimd_vdot_f32c_accurate);
#endif

#if SIMSIMD_TARGET_SVE
    register_<simsimd_f16_t>("dot_f16_sve", simsimd_dot_f16_sve, simsimd_dot_f16_accurate);
    register_<simsimd_f16_t>("cos_f16_sve", simsimd_cos_f16_sve, simsimd_cos_f16_accurate);
    register_<simsimd_f16_t>("l2sq_f16_sve", simsimd_l2sq_f16_sve, simsimd_l2sq_f16_accurate);

    register_<simsimd_f32_t>("dot_f32_sve", simsimd_dot_f32_sve, simsimd_dot_f32_accurate);
    register_<simsimd_f32_t>("cos_f32_sve", simsimd_cos_f32_sve, simsimd_cos_f32_accurate);
    register_<simsimd_f32_t>("l2sq_f32_sve", simsimd_l2sq_f32_sve, simsimd_l2sq_f32_accurate);

    register_<simsimd_f64_t>("dot_f64_sve", simsimd_dot_f64_sve, simsimd_dot_f64_serial);
    register_<simsimd_f64_t>("cos_f64_sve", simsimd_cos_f64_sve, simsimd_cos_f64_serial);
    register_<simsimd_f64_t>("l2sq_f64_sve", simsimd_l2sq_f64_sve, simsimd_l2sq_f64_serial);

    register_<simsimd_b8_t>("hamming_b8_sve", simsimd_hamming_b8_sve, simsimd_hamming_b8_serial);
    register_<simsimd_b8_t>("jaccard_b8_sve", simsimd_jaccard_b8_sve, simsimd_jaccard_b8_serial);

    register_<simsimd_f16_t>("dot_f16c_sve", simsimd_dot_f16c_sve, simsimd_dot_f16c_accurate);
    register_<simsimd_f16_t>("vdot_f16c_sve", simsimd_vdot_f16c_sve, simsimd_vdot_f16c_accurate);
    register_<simsimd_f32_t>("dot_f32c_sve", simsimd_dot_f32c_sve, simsimd_dot_f32c_accurate);
    register_<simsimd_f32_t>("vdot_f32c_sve", simsimd_vdot_f32c_sve, simsimd_vdot_f32c_accurate);
    register_<simsimd_f64_t>("dot_f64c_sve", simsimd_dot_f64c_sve, simsimd_dot_f64c_serial);
    register_<simsimd_f64_t>("vdot_f64c_sve", simsimd_vdot_f64c_sve, simsimd_vdot_f64c_serial);

#endif

#if SIMSIMD_TARGET_HASWELL
    register_<simsimd_f16_t>("dot_f16_haswell", simsimd_dot_f16_haswell, simsimd_dot_f16_accurate);
    register_<simsimd_f16_t>("cos_f16_haswell", simsimd_cos_f16_haswell, simsimd_cos_f16_accurate);
    register_<simsimd_f16_t>("l2sq_f16_haswell", simsimd_l2sq_f16_haswell, simsimd_l2sq_f16_accurate);
    register_<simsimd_f16_t>("kl_f16_haswell", simsimd_kl_f16_haswell, simsimd_kl_f16_accurate);
    register_<simsimd_f16_t>("js_f16_haswell", simsimd_js_f16_haswell, simsimd_js_f16_accurate);

    register_<simsimd_i8_t>("cos_i8_haswell", simsimd_cos_i8_haswell, simsimd_cos_i8_accurate);
    register_<simsimd_i8_t>("l2sq_i8_haswell", simsimd_l2sq_i8_haswell, simsimd_l2sq_i8_accurate);

    register_<simsimd_b8_t>("hamming_b8_haswell", simsimd_hamming_b8_haswell, simsimd_hamming_b8_serial);
    register_<simsimd_b8_t>("jaccard_b8_haswell", simsimd_jaccard_b8_haswell, simsimd_jaccard_b8_serial);

    register_<simsimd_f16_t>("dot_f16c_haswell", simsimd_dot_f16c_haswell, simsimd_dot_f16c_accurate);
    register_<simsimd_f16_t>("vdot_f16c_haswell", simsimd_vdot_f16c_haswell, simsimd_vdot_f16c_accurate);
    register_<simsimd_f32_t>("dot_f32c_haswell", simsimd_dot_f32c_haswell, simsimd_dot_f32c_accurate);
    register_<simsimd_f32_t>("vdot_f32c_haswell", simsimd_vdot_f32c_haswell, simsimd_vdot_f32c_accurate);
#endif

#if SIMSIMD_TARGET_SAPPHIRE
    register_<simsimd_f16_t>("dot_f16_sapphire", simsimd_dot_f16_sapphire, simsimd_dot_f16_accurate);
    register_<simsimd_f16_t>("cos_f16_sapphire", simsimd_cos_f16_sapphire, simsimd_cos_f16_accurate);
    register_<simsimd_f16_t>("l2sq_f16_sapphire", simsimd_l2sq_f16_sapphire, simsimd_l2sq_f16_accurate);
    register_<simsimd_f16_t>("kl_f16_sapphire", simsimd_kl_f16_sapphire, simsimd_kl_f16_accurate);
    register_<simsimd_f16_t>("js_f16_sapphire", simsimd_js_f16_sapphire, simsimd_js_f16_accurate);

    register_<simsimd_f16_t>("dot_f16c_sapphire", simsimd_dot_f16c_sapphire, simsimd_dot_f16c_accurate);
    register_<simsimd_f16_t>("vdot_f16c_sapphire", simsimd_vdot_f16c_sapphire, simsimd_vdot_f16c_accurate);
#endif

#if SIMSIMD_TARGET_ICE
    register_<simsimd_i8_t>("cos_i8_ice", simsimd_cos_i8_ice, simsimd_cos_i8_accurate);
    register_<simsimd_i8_t>("l2sq_i8_ice", simsimd_l2sq_i8_ice, simsimd_l2sq_i8_accurate);

    register_<simsimd_f64_t>("dot_f64_skylake", simsimd_dot_f64_skylake, simsimd_dot_f64_serial);
    register_<simsimd_f64_t>("cos_f64_skylake", simsimd_cos_f64_skylake, simsimd_cos_f64_serial);
    register_<simsimd_f64_t>("l2sq_f64_skylake", simsimd_l2sq_f64_skylake, simsimd_l2sq_f64_serial);

    register_<simsimd_b8_t>("hamming_b8_ice", simsimd_hamming_b8_ice, simsimd_hamming_b8_serial);
    register_<simsimd_b8_t>("jaccard_b8_ice", simsimd_jaccard_b8_ice, simsimd_jaccard_b8_serial);
#endif

#if SIMSIMD_TARGET_SKYLAKE
    register_<simsimd_f32_t>("dot_f32_skylake", simsimd_dot_f32_skylake, simsimd_dot_f32_accurate);
    register_<simsimd_f32_t>("cos_f32_skylake", simsimd_cos_f32_skylake, simsimd_cos_f32_accurate);
    register_<simsimd_f32_t>("l2sq_f32_skylake", simsimd_l2sq_f32_skylake, simsimd_l2sq_f32_accurate);
    register_<simsimd_f32_t>("kl_f32_skylake", simsimd_kl_f32_skylake, simsimd_kl_f32_accurate);
    register_<simsimd_f32_t>("js_f32_skylake", simsimd_js_f32_skylake, simsimd_js_f32_accurate);

    register_<simsimd_f32_t>("dot_f32c_skylake", simsimd_dot_f32c_skylake, simsimd_dot_f32c_accurate);
    register_<simsimd_f32_t>("vdot_f32c_skylake", simsimd_vdot_f32c_skylake, simsimd_vdot_f32c_accurate);
    register_<simsimd_f64_t>("dot_f64c_skylake", simsimd_dot_f64c_skylake, simsimd_dot_f64c_serial);
    register_<simsimd_f64_t>("vdot_f64c_skylake", simsimd_vdot_f64c_skylake, simsimd_vdot_f64c_serial);
#endif

    register_<simsimd_f16_t>("dot_f16_serial", simsimd_dot_f16_serial, simsimd_dot_f16_accurate);
    register_<simsimd_f16_t>("cos_f16_serial", simsimd_cos_f16_serial, simsimd_cos_f16_accurate);
    register_<simsimd_f16_t>("l2sq_f16_serial", simsimd_l2sq_f16_serial, simsimd_l2sq_f16_accurate);
    register_<simsimd_f16_t>("kl_f16_serial", simsimd_kl_f16_serial, simsimd_kl_f16_accurate);
    register_<simsimd_f16_t>("js_f16_serial", simsimd_js_f16_serial, simsimd_js_f16_accurate);

    register_<simsimd_f32_t>("dot_f32_serial", simsimd_dot_f32_serial, simsimd_dot_f32_accurate);
    register_<simsimd_f32_t>("cos_f32_serial", simsimd_cos_f32_serial, simsimd_cos_f32_accurate);
    register_<simsimd_f32_t>("l2sq_f32_serial", simsimd_l2sq_f32_serial, simsimd_l2sq_f32_accurate);
    register_<simsimd_f32_t>("kl_f32_serial", simsimd_kl_f32_serial, simsimd_kl_f32_accurate);
    register_<simsimd_f32_t>("js_f32_serial", simsimd_js_f32_serial, simsimd_js_f32_accurate);

    register_<simsimd_f64_t>("dot_f64_serial", simsimd_dot_f64_serial, simsimd_dot_f64_serial);
    register_<simsimd_f64_t>("cos_f64_serial", simsimd_cos_f64_serial, simsimd_cos_f64_serial);
    register_<simsimd_f64_t>("l2sq_f64_serial", simsimd_l2sq_f64_serial, simsimd_l2sq_f64_serial);

    register_<simsimd_i8_t>("cos_i8_serial", simsimd_cos_i8_serial, simsimd_cos_i8_accurate);
    register_<simsimd_i8_t>("l2sq_i8_serial", simsimd_l2sq_i8_serial, simsimd_l2sq_i8_accurate);

    register_<simsimd_f64_t>("dot_f64c_serial", simsimd_dot_f64c_serial, simsimd_dot_f64c_serial);
    register_<simsimd_f32_t>("dot_f32c_serial", simsimd_dot_f32c_serial, simsimd_dot_f32c_accurate);
    register_<simsimd_f16_t>("dot_f16c_serial", simsimd_dot_f16c_serial, simsimd_dot_f16c_accurate);

    register_<simsimd_b8_t>("hamming_b8_serial", simsimd_hamming_b8_serial, simsimd_hamming_b8_serial);
    register_<simsimd_b8_t>("jaccard_b8_serial", simsimd_jaccard_b8_serial, simsimd_jaccard_b8_serial);

    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}