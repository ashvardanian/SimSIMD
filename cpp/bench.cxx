#include <cmath>   // `std::sqrt`
#include <cstring> // `std::memcpy`
#include <thread>  // `std::thread`

#include <benchmark/benchmark.h>

#if !defined(SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS)
#define SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS 0
#endif
#if SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS
#include <cblas.h>
#endif

// It's important to note, that out compression/decompression routines
// are quite inaccurate. They are not meant to be used in production code.
// So when benchmarking, if possible, please use the native types, if those
// are implemeneted.
#define SIMSIMD_NATIVE_F16 1
#define SIMSIMD_NATIVE_BF16 1
#define SIMSIMD_RSQRT(x) (1 / sqrtf(x))
#define SIMSIMD_LOG(x) (logf(x))
#include <simsimd/simsimd.h>

namespace bm = benchmark;

// clang-format off
template <simsimd_datatype_t> struct datatype_enum_to_type_gt { using value_t = void; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_f64_k> { using value_t = simsimd_f64_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_f32_k> { using value_t = simsimd_f32_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_f16_k> { using value_t = simsimd_f16_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_bf16_k> { using value_t = simsimd_bf16_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_i8_k> { using value_t = simsimd_i8_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_b8_k> { using value_t = simsimd_b8_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_f64c_k> { using value_t = simsimd_f64_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_f32c_k> { using value_t = simsimd_f32_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_f16c_k> { using value_t = simsimd_f16_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_bf16c_k> { using value_t = simsimd_bf16_t; };
// clang-format on

template <simsimd_datatype_t datatype_ak, std::size_t dimensions_ak> struct vectors_pair_gt {
    using scalar_t = typename datatype_enum_to_type_gt<datatype_ak>::value_t;
    using compressed16_t = unsigned short;

    scalar_t a[dimensions_ak]{};
    scalar_t b[dimensions_ak]{};

    std::size_t dimensions() const noexcept { return dimensions_ak; }
    std::size_t size_bytes() const noexcept { return dimensions_ak * sizeof(scalar_t); }

    void set(scalar_t v) noexcept {
        for (std::size_t i = 0; i != dimensions_ak; ++i)
            a[i] = b[i] = v;
    }

    void compress(double const& from, scalar_t& to) noexcept {
#if !SIMSIMD_NATIVE_BF16
        if constexpr (datatype_ak == simsimd_datatype_bf16_k || datatype_ak == simsimd_datatype_bf16c_k) {
            auto compressed = simsimd_compress_bf16(from);
            std::memcpy(&to, &compressed, sizeof(scalar_t));
            static_assert(sizeof(scalar_t) == sizeof(compressed));
            return;
        }
#endif
#if !SIMSIMD_NATIVE_F16
        if constexpr (datatype_ak == simsimd_datatype_f16_k || datatype_ak == simsimd_datatype_f16c_k) {
            auto compressed = simsimd_compress_f16(from);
            std::memcpy(&to, &compressed, sizeof(scalar_t));
            static_assert(sizeof(scalar_t) == sizeof(compressed));
            return;
        }
#endif
        to = static_cast<scalar_t>(from);
    }

    double uncompress(scalar_t const& from) noexcept {
#if !SIMSIMD_NATIVE_BF16
        if constexpr (datatype_ak == simsimd_datatype_bf16_k || datatype_ak == simsimd_datatype_bf16c_k) {
            compressed16_t compressed;
            std::memcpy(&compressed, &from, sizeof(scalar_t));
            return simsimd_uncompress_bf16(compressed);
        }
#endif
#if !SIMSIMD_NATIVE_F16
        if constexpr (datatype_ak == simsimd_datatype_f16_k || datatype_ak == simsimd_datatype_f16c_k) {
            compressed16_t compressed;
            std::memcpy(&compressed, &from, sizeof(scalar_t));
            return simsimd_uncompress_f16(compressed);
        }
#endif
        return from;
    }

    void randomize() noexcept {

        double a2_sum = 0, b2_sum = 0;
        for (std::size_t i = 0; i != dimensions_ak; ++i) {
            if constexpr (std::is_integral<scalar_t>())
                a[i] = static_cast<scalar_t>(std::rand() % std::numeric_limits<scalar_t>::max()),
                b[i] = static_cast<scalar_t>(std::rand() % std::numeric_limits<scalar_t>::max());
            else {
                double ai = double(std::rand()) / double(RAND_MAX), bi = double(std::rand()) / double(RAND_MAX);
                a2_sum += ai * ai, b2_sum += bi * bi;
                compress(ai, a[i]), compress(bi, b[i]);
            }
        }

        // Normalize the vectors:
        if constexpr (!std::is_integral<scalar_t>()) {
            a2_sum = std::sqrt(a2_sum);
            b2_sum = std::sqrt(b2_sum);
            for (std::size_t i = 0; i != dimensions_ak; ++i)
                compress(uncompress(a[i]) / a2_sum, a[i]), compress(uncompress(b[i]) / b2_sum, b[i]);
        }
    }
};

template <typename pair_at, typename metric_at = void>
void measure(bm::State& state, metric_at metric, metric_at baseline) {

    auto call_baseline = [&](pair_at& pair) -> double {
        // Output for real vectors have a single dimensions.
        // Output for complex vectors have two dimensions.
        simsimd_distance_t results[2] = {0, 0};
        baseline(pair.a, pair.b, pair.dimensions(), &results[0]);
        return results[0] + results[1];
    };
    auto call_contender = [&](pair_at& pair) -> double {
        // Output for real vectors have a single dimensions.
        // Output for complex vectors have two dimensions.
        simsimd_distance_t results[2] = {0, 0};
        metric(pair.a, pair.b, pair.dimensions(), &results[0]);
        return results[0] + results[1];
    };

    // Let's average the distance results over many pairs.
    std::vector<pair_at> pairs(1024);
    for (auto& pair : pairs)
        pair.randomize();

    // Initialize the output buffers for distance calculations.
    std::vector<double> results_baseline(pairs.size());
    std::vector<double> results_contender(pairs.size());
    for (std::size_t i = 0; i != pairs.size(); ++i)
        results_baseline[i] = call_baseline(pairs[i]), results_contender[i] = call_contender(pairs[i]);

    // The actual benchmarking loop.
    std::size_t iterations = 0;
    for (auto _ : state)
        bm::DoNotOptimize((results_contender[iterations & 1023] = call_contender(pairs[iterations & 1023]))),
            iterations++;

    // Measure the mean absolute delta and relative error.
    double mean_delta = 0, mean_relative_error = 0;
    for (std::size_t i = 0; i != pairs.size(); ++i) {
        auto abs_delta = std::abs(results_contender[i] - results_baseline[i]);
        mean_delta += abs_delta;
        double error = abs_delta != 0 && results_baseline[i] != 0 ? abs_delta / results_baseline[i] : 0;
        mean_relative_error += error;
    }
    mean_delta /= pairs.size();
    mean_relative_error /= pairs.size();
    state.counters["abs_delta"] = mean_delta;
    state.counters["relative_error"] = mean_relative_error;
    state.counters["bytes"] = bm::Counter(iterations * pairs[0].size_bytes() * 2, bm::Counter::kIsRate);
    state.counters["pairs"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <simsimd_datatype_t datatype_ak, typename metric_at = void>
void register_(std::string name, metric_at* distance_func, metric_at* baseline_func) {

    std::size_t seconds = 10;
    std::size_t threads = 1;

    using pair_dims_t = vectors_pair_gt<datatype_ak, 1536>;
    using scalar_t = typename pair_dims_t::scalar_t;
    using pair_bytes_t = vectors_pair_gt<datatype_ak, 1536 / sizeof(scalar_t)>;

    std::string name_dims = name + "_" + std::to_string(pair_dims_t{}.dimensions()) + "d";
    bm::RegisterBenchmark(name_dims.c_str(), measure<pair_dims_t, metric_at*>, distance_func, baseline_func)
        ->MinTime(seconds)
        ->Threads(threads);

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
    std::printf("- Compiler used native BF16: %s\n", flags[SIMSIMD_NATIVE_BF16]);
    std::printf("- Benchmark against CBLAS: %s\n", flags[SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS]);
    std::printf("\n");
    std::printf("Compile-time settings:\n");
    std::printf("- Arm NEON support enabled: %s\n", flags[SIMSIMD_TARGET_NEON]);
    std::printf("- Arm SVE support enabled: %s\n", flags[SIMSIMD_TARGET_SVE]);
    std::printf("- x86 Haswell support enabled: %s\n", flags[SIMSIMD_TARGET_HASWELL]);
    std::printf("- x86 Skylake support enabled: %s\n", flags[SIMSIMD_TARGET_SKYLAKE]);
    std::printf("- x86 Ice Lake support enabled: %s\n", flags[SIMSIMD_TARGET_ICE]);
    std::printf("- x86 Genoa support enabled: %s\n", flags[SIMSIMD_TARGET_GENOA]);
    std::printf("- x86 Sapphire Rapids support enabled: %s\n", flags[SIMSIMD_TARGET_SAPPHIRE]);
    std::printf("\n");
    std::printf("Run-time settings:\n");
    std::printf("- Arm NEON support enabled: %s\n", flags[(runtime_caps & simsimd_cap_neon_k) != 0]);
    std::printf("- Arm SVE support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sve_k) != 0]);
    std::printf("- x86 Haswell support enabled: %s\n", flags[(runtime_caps & simsimd_cap_haswell_k) != 0]);
    std::printf("- x86 Skylake support enabled: %s\n", flags[(runtime_caps & simsimd_cap_skylake_k) != 0]);
    std::printf("- x86 Ice Lake support enabled: %s\n", flags[(runtime_caps & simsimd_cap_ice_k) != 0]);
    std::printf("- x86 Genoa support enabled: %s\n", flags[(runtime_caps & simsimd_cap_genoa_k) != 0]);
    std::printf("- x86 Sapphire Rapids support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sapphire_k) != 0]);
    std::printf("\n");

    // Run the benchmarks
    bm::Initialize(&argc, argv);
    if (bm::ReportUnrecognizedArguments(argc, argv))
        return 1;

#if SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS

    register_<simsimd_datatype_f32_k>("dot_f32_blas", dot_f32_blas, simsimd_dot_f32_accurate);
    register_<simsimd_datatype_f64_k>("dot_f64_blas", dot_f64_blas, simsimd_dot_f64_serial);
    register_<simsimd_datatype_f32c_k>("dot_f32c_blas", dot_f32c_blas, simsimd_dot_f32c_accurate);
    register_<simsimd_datatype_f64c_k>("dot_f64c_blas", dot_f64c_blas, simsimd_dot_f64c_serial);
    register_<simsimd_datatype_f32c_k>("vdot_f32c_blas", vdot_f32c_blas, simsimd_vdot_f32c_accurate);
    register_<simsimd_datatype_f64c_k>("vdot_f64c_blas", vdot_f64c_blas, simsimd_vdot_f64c_serial);

#endif

#if SIMSIMD_TARGET_NEON

    register_<simsimd_datatype_f16_k>("dot_f16_neon", simsimd_dot_f16_neon, simsimd_dot_f16_accurate);
    register_<simsimd_datatype_f16_k>("cos_f16_neon", simsimd_cos_f16_neon, simsimd_cos_f16_accurate);
    register_<simsimd_datatype_f16_k>("l2sq_f16_neon", simsimd_l2sq_f16_neon, simsimd_l2sq_f16_accurate);
    register_<simsimd_datatype_f16_k>("kl_f16_neon", simsimd_kl_f16_neon, simsimd_kl_f16_accurate);
    register_<simsimd_datatype_f16_k>("js_f16_neon", simsimd_js_f16_neon, simsimd_js_f16_accurate);

    register_<simsimd_datatype_f32_k>("dot_f32_neon", simsimd_dot_f32_neon, simsimd_dot_f32_accurate);
    register_<simsimd_datatype_f32_k>("cos_f32_neon", simsimd_cos_f32_neon, simsimd_cos_f32_accurate);
    register_<simsimd_datatype_f32_k>("l2sq_f32_neon", simsimd_l2sq_f32_neon, simsimd_l2sq_f32_accurate);
    register_<simsimd_datatype_f32_k>("kl_f32_neon", simsimd_kl_f32_neon, simsimd_kl_f32_accurate);
    register_<simsimd_datatype_f32_k>("js_f32_neon", simsimd_js_f32_neon, simsimd_js_f32_accurate);

    register_<simsimd_datatype_i8_k>("cos_i8_neon", simsimd_cos_i8_neon, simsimd_cos_i8_accurate);
    register_<simsimd_datatype_i8_k>("dot_i8_neon", simsimd_dot_i8_neon, simsimd_dot_i8_serial);
    register_<simsimd_datatype_i8_k>("l2sq_i8_neon", simsimd_l2sq_i8_neon, simsimd_l2sq_i8_accurate);

    register_<simsimd_datatype_b8_k>("hamming_b8_neon", simsimd_hamming_b8_neon, simsimd_hamming_b8_serial);
    register_<simsimd_datatype_b8_k>("jaccard_b8_neon", simsimd_jaccard_b8_neon, simsimd_jaccard_b8_serial);

    register_<simsimd_datatype_f16c_k>("dot_f16c_neon", simsimd_dot_f16c_neon, simsimd_dot_f16c_accurate);
    register_<simsimd_datatype_f16c_k>("vdot_f16c_neon", simsimd_vdot_f16c_neon, simsimd_vdot_f16c_accurate);
    register_<simsimd_datatype_f32c_k>("dot_f32c_neon", simsimd_dot_f32c_neon, simsimd_dot_f32c_accurate);
    register_<simsimd_datatype_f32c_k>("vdot_f32c_neon", simsimd_vdot_f32c_neon, simsimd_vdot_f32c_accurate);
#endif

#if SIMSIMD_TARGET_SVE
    register_<simsimd_datatype_f16_k>("dot_f16_sve", simsimd_dot_f16_sve, simsimd_dot_f16_accurate);
    register_<simsimd_datatype_f16_k>("cos_f16_sve", simsimd_cos_f16_sve, simsimd_cos_f16_accurate);
    register_<simsimd_datatype_f16_k>("l2sq_f16_sve", simsimd_l2sq_f16_sve, simsimd_l2sq_f16_accurate);

    register_<simsimd_datatype_f32_k>("dot_f32_sve", simsimd_dot_f32_sve, simsimd_dot_f32_accurate);
    register_<simsimd_datatype_f32_k>("cos_f32_sve", simsimd_cos_f32_sve, simsimd_cos_f32_accurate);
    register_<simsimd_datatype_f32_k>("l2sq_f32_sve", simsimd_l2sq_f32_sve, simsimd_l2sq_f32_accurate);

    register_<simsimd_datatype_f64_k>("dot_f64_sve", simsimd_dot_f64_sve, simsimd_dot_f64_serial);
    register_<simsimd_datatype_f64_k>("cos_f64_sve", simsimd_cos_f64_sve, simsimd_cos_f64_serial);
    register_<simsimd_datatype_f64_k>("l2sq_f64_sve", simsimd_l2sq_f64_sve, simsimd_l2sq_f64_serial);

    register_<simsimd_datatype_b8_k>("hamming_b8_sve", simsimd_hamming_b8_sve, simsimd_hamming_b8_serial);
    register_<simsimd_datatype_b8_k>("jaccard_b8_sve", simsimd_jaccard_b8_sve, simsimd_jaccard_b8_serial);

    register_<simsimd_datatype_f16c_k>("dot_f16c_sve", simsimd_dot_f16c_sve, simsimd_dot_f16c_accurate);
    register_<simsimd_datatype_f16c_k>("vdot_f16c_sve", simsimd_vdot_f16c_sve, simsimd_vdot_f16c_accurate);
    register_<simsimd_datatype_f32c_k>("dot_f32c_sve", simsimd_dot_f32c_sve, simsimd_dot_f32c_accurate);
    register_<simsimd_datatype_f32c_k>("vdot_f32c_sve", simsimd_vdot_f32c_sve, simsimd_vdot_f32c_accurate);
    register_<simsimd_datatype_f64c_k>("dot_f64c_sve", simsimd_dot_f64c_sve, simsimd_dot_f64c_serial);
    register_<simsimd_datatype_f64c_k>("vdot_f64c_sve", simsimd_vdot_f64c_sve, simsimd_vdot_f64c_serial);

#endif

#if SIMSIMD_TARGET_HASWELL
    register_<simsimd_datatype_f16_k>("dot_f16_haswell", simsimd_dot_f16_haswell, simsimd_dot_f16_accurate);
    register_<simsimd_datatype_f16_k>("cos_f16_haswell", simsimd_cos_f16_haswell, simsimd_cos_f16_accurate);
    register_<simsimd_datatype_f16_k>("l2sq_f16_haswell", simsimd_l2sq_f16_haswell, simsimd_l2sq_f16_accurate);
    register_<simsimd_datatype_f16_k>("kl_f16_haswell", simsimd_kl_f16_haswell, simsimd_kl_f16_accurate);
    register_<simsimd_datatype_f16_k>("js_f16_haswell", simsimd_js_f16_haswell, simsimd_js_f16_accurate);

    register_<simsimd_datatype_bf16_k>("dot_bf16_haswell", simsimd_dot_bf16_haswell, simsimd_dot_bf16_accurate);
    register_<simsimd_datatype_bf16_k>("cos_bf16_haswell", simsimd_cos_bf16_haswell, simsimd_cos_bf16_accurate);
    register_<simsimd_datatype_bf16_k>("l2sq_bf16_haswell", simsimd_l2sq_bf16_haswell, simsimd_l2sq_bf16_accurate);

    register_<simsimd_datatype_i8_k>("cos_i8_haswell", simsimd_cos_i8_haswell, simsimd_cos_i8_accurate);
    register_<simsimd_datatype_i8_k>("dot_i8_haswell", simsimd_dot_i8_haswell, simsimd_dot_i8_serial);
    register_<simsimd_datatype_i8_k>("l2sq_i8_haswell", simsimd_l2sq_i8_haswell, simsimd_l2sq_i8_accurate);

    register_<simsimd_datatype_b8_k>("hamming_b8_haswell", simsimd_hamming_b8_haswell, simsimd_hamming_b8_serial);
    register_<simsimd_datatype_b8_k>("jaccard_b8_haswell", simsimd_jaccard_b8_haswell, simsimd_jaccard_b8_serial);

    register_<simsimd_datatype_f16c_k>("dot_f16c_haswell", simsimd_dot_f16c_haswell, simsimd_dot_f16c_accurate);
    register_<simsimd_datatype_f16c_k>("vdot_f16c_haswell", simsimd_vdot_f16c_haswell, simsimd_vdot_f16c_accurate);
    register_<simsimd_datatype_f32c_k>("dot_f32c_haswell", simsimd_dot_f32c_haswell, simsimd_dot_f32c_accurate);
    register_<simsimd_datatype_f32c_k>("vdot_f32c_haswell", simsimd_vdot_f32c_haswell, simsimd_vdot_f32c_accurate);
#endif

#if SIMSIMD_TARGET_GENOA
    register_<simsimd_datatype_bf16_k>("dot_bf16_genoa", simsimd_dot_bf16_genoa, simsimd_dot_bf16_accurate);
    register_<simsimd_datatype_bf16_k>("cos_bf16_genoa", simsimd_cos_bf16_genoa, simsimd_cos_bf16_accurate);
    register_<simsimd_datatype_bf16_k>("l2sq_bf16_genoa", simsimd_l2sq_bf16_genoa, simsimd_l2sq_bf16_accurate);
#endif

#if SIMSIMD_TARGET_SAPPHIRE
    register_<simsimd_datatype_f16_k>("dot_f16_sapphire", simsimd_dot_f16_sapphire, simsimd_dot_f16_accurate);
    register_<simsimd_datatype_f16_k>("cos_f16_sapphire", simsimd_cos_f16_sapphire, simsimd_cos_f16_accurate);
    register_<simsimd_datatype_f16_k>("l2sq_f16_sapphire", simsimd_l2sq_f16_sapphire, simsimd_l2sq_f16_accurate);
    register_<simsimd_datatype_f16_k>("kl_f16_sapphire", simsimd_kl_f16_sapphire, simsimd_kl_f16_accurate);
    register_<simsimd_datatype_f16_k>("js_f16_sapphire", simsimd_js_f16_sapphire, simsimd_js_f16_accurate);

    register_<simsimd_datatype_f16c_k>("dot_f16c_sapphire", simsimd_dot_f16c_sapphire, simsimd_dot_f16c_accurate);
    register_<simsimd_datatype_f16c_k>("vdot_f16c_sapphire", simsimd_vdot_f16c_sapphire, simsimd_vdot_f16c_accurate);
#endif

#if SIMSIMD_TARGET_ICE
    register_<simsimd_datatype_i8_k>("cos_i8_ice", simsimd_cos_i8_ice, simsimd_cos_i8_accurate);
    register_<simsimd_datatype_i8_k>("dot_i8_ice", simsimd_dot_i8_ice, simsimd_dot_i8_serial);
    register_<simsimd_datatype_i8_k>("l2sq_i8_ice", simsimd_l2sq_i8_ice, simsimd_l2sq_i8_accurate);

    register_<simsimd_datatype_f64_k>("dot_f64_skylake", simsimd_dot_f64_skylake, simsimd_dot_f64_serial);
    register_<simsimd_datatype_f64_k>("cos_f64_skylake", simsimd_cos_f64_skylake, simsimd_cos_f64_serial);
    register_<simsimd_datatype_f64_k>("l2sq_f64_skylake", simsimd_l2sq_f64_skylake, simsimd_l2sq_f64_serial);

    register_<simsimd_datatype_b8_k>("hamming_b8_ice", simsimd_hamming_b8_ice, simsimd_hamming_b8_serial);
    register_<simsimd_datatype_b8_k>("jaccard_b8_ice", simsimd_jaccard_b8_ice, simsimd_jaccard_b8_serial);
#endif

#if SIMSIMD_TARGET_SKYLAKE
    register_<simsimd_datatype_f32_k>("dot_f32_skylake", simsimd_dot_f32_skylake, simsimd_dot_f32_accurate);
    register_<simsimd_datatype_f32_k>("cos_f32_skylake", simsimd_cos_f32_skylake, simsimd_cos_f32_accurate);
    register_<simsimd_datatype_f32_k>("l2sq_f32_skylake", simsimd_l2sq_f32_skylake, simsimd_l2sq_f32_accurate);
    register_<simsimd_datatype_f32_k>("kl_f32_skylake", simsimd_kl_f32_skylake, simsimd_kl_f32_accurate);
    register_<simsimd_datatype_f32_k>("js_f32_skylake", simsimd_js_f32_skylake, simsimd_js_f32_accurate);

    register_<simsimd_datatype_f32c_k>("dot_f32c_skylake", simsimd_dot_f32c_skylake, simsimd_dot_f32c_accurate);
    register_<simsimd_datatype_f32c_k>("vdot_f32c_skylake", simsimd_vdot_f32c_skylake, simsimd_vdot_f32c_accurate);
    register_<simsimd_datatype_f64c_k>("dot_f64c_skylake", simsimd_dot_f64c_skylake, simsimd_dot_f64c_serial);
    register_<simsimd_datatype_f64c_k>("vdot_f64c_skylake", simsimd_vdot_f64c_skylake, simsimd_vdot_f64c_serial);
#endif

    register_<simsimd_datatype_bf16_k>("dot_bf16_serial", simsimd_dot_bf16_serial, simsimd_dot_bf16_accurate);
    register_<simsimd_datatype_bf16_k>("cos_bf16_serial", simsimd_cos_bf16_serial, simsimd_cos_bf16_accurate);
    register_<simsimd_datatype_bf16_k>("l2sq_bf16_serial", simsimd_l2sq_bf16_serial, simsimd_l2sq_bf16_accurate);
    register_<simsimd_datatype_bf16_k>("kl_bf16_serial", simsimd_kl_bf16_serial, simsimd_kl_bf16_accurate);
    register_<simsimd_datatype_bf16_k>("js_bf16_serial", simsimd_js_bf16_serial, simsimd_js_bf16_accurate);

    register_<simsimd_datatype_f16_k>("dot_f16_serial", simsimd_dot_f16_serial, simsimd_dot_f16_accurate);
    register_<simsimd_datatype_f16_k>("cos_f16_serial", simsimd_cos_f16_serial, simsimd_cos_f16_accurate);
    register_<simsimd_datatype_f16_k>("l2sq_f16_serial", simsimd_l2sq_f16_serial, simsimd_l2sq_f16_accurate);
    register_<simsimd_datatype_f16_k>("kl_f16_serial", simsimd_kl_f16_serial, simsimd_kl_f16_accurate);
    register_<simsimd_datatype_f16_k>("js_f16_serial", simsimd_js_f16_serial, simsimd_js_f16_accurate);

    register_<simsimd_datatype_f32_k>("dot_f32_serial", simsimd_dot_f32_serial, simsimd_dot_f32_accurate);
    register_<simsimd_datatype_f32_k>("cos_f32_serial", simsimd_cos_f32_serial, simsimd_cos_f32_accurate);
    register_<simsimd_datatype_f32_k>("l2sq_f32_serial", simsimd_l2sq_f32_serial, simsimd_l2sq_f32_accurate);
    register_<simsimd_datatype_f32_k>("kl_f32_serial", simsimd_kl_f32_serial, simsimd_kl_f32_accurate);
    register_<simsimd_datatype_f32_k>("js_f32_serial", simsimd_js_f32_serial, simsimd_js_f32_accurate);

    register_<simsimd_datatype_f64_k>("dot_f64_serial", simsimd_dot_f64_serial, simsimd_dot_f64_serial);
    register_<simsimd_datatype_f64_k>("cos_f64_serial", simsimd_cos_f64_serial, simsimd_cos_f64_serial);
    register_<simsimd_datatype_f64_k>("l2sq_f64_serial", simsimd_l2sq_f64_serial, simsimd_l2sq_f64_serial);

    register_<simsimd_datatype_i8_k>("cos_i8_serial", simsimd_cos_i8_serial, simsimd_cos_i8_accurate);
    register_<simsimd_datatype_i8_k>("dot_i8_serial", simsimd_dot_i8_serial, simsimd_dot_i8_serial);
    register_<simsimd_datatype_i8_k>("l2sq_i8_serial", simsimd_l2sq_i8_serial, simsimd_l2sq_i8_accurate);

    register_<simsimd_datatype_f64c_k>("dot_f64c_serial", simsimd_dot_f64c_serial, simsimd_dot_f64c_serial);
    register_<simsimd_datatype_f32c_k>("dot_f32c_serial", simsimd_dot_f32c_serial, simsimd_dot_f32c_accurate);
    register_<simsimd_datatype_f16c_k>("dot_f16c_serial", simsimd_dot_f16c_serial, simsimd_dot_f16c_accurate);
    register_<simsimd_datatype_bf16c_k>("dot_bf16c_serial", simsimd_dot_bf16c_serial, simsimd_dot_bf16c_accurate);

    register_<simsimd_datatype_f16c_k>("vdot_f16c_serial", simsimd_vdot_f16c_serial, simsimd_vdot_f16c_accurate);
    register_<simsimd_datatype_bf16c_k>("vdot_bf16c_serial", simsimd_vdot_bf16c_serial, simsimd_vdot_bf16c_accurate);

    register_<simsimd_datatype_b8_k>("hamming_b8_serial", simsimd_hamming_b8_serial, simsimd_hamming_b8_serial);
    register_<simsimd_datatype_b8_k>("jaccard_b8_serial", simsimd_jaccard_b8_serial, simsimd_jaccard_b8_serial);

    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}
