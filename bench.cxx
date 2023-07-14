#include <thread>

#include <benchmark/benchmark.h>

#include <simsimd/simsimd.h>
#include <simsimd/simsimd_chem.h>

namespace bm = benchmark;

static const std::size_t threads_k = std::thread::hardware_concurrency();
static constexpr std::size_t time_k = 10;
static constexpr std::size_t bytes_k = 1024;

template <typename return_at, typename... args_at>
constexpr std::size_t number_of_arguments(return_at (*f)(args_at...)) {
    return sizeof...(args_at);
}

template <typename scalar_at, std::size_t bytes_per_vector_ak = bytes_k,
          typename metric_at = void> //
static void measure(bm::State& state, metric_at metric) {

    constexpr ::size_t dimensions_ak = bytes_per_vector_ak / sizeof(scalar_at);
    alignas(64) scalar_at a[dimensions_ak]{};
    alignas(64) scalar_at b[dimensions_ak]{};
    float c{};

    std::fill_n(a, dimensions_ak, static_cast<scalar_at>(1));
    std::fill_n(b, dimensions_ak, static_cast<scalar_at>(2));

    std::size_t iterations = 0;
    for (auto _ : state)
        if constexpr (number_of_arguments(metric_at{}) == 3)
            bm::DoNotOptimize((c = metric(a, b, dimensions_ak))), iterations++;
        else
            bm::DoNotOptimize((c = metric(a, b))), iterations++;

    state.counters["bytes"] = bm::Counter(iterations * bytes_per_vector_ak * 2u, bm::Counter::kIsRate);
    state.counters["pairs"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <typename scalar_at, std::size_t bytes_per_vector_ak = bytes_k, typename metric_at = void>
void register_(char const* name, metric_at distance_func) {
    bm::RegisterBenchmark(name, measure<scalar_at, bytes_per_vector_ak, metric_at>, distance_func)
        ->Threads(1)
        ->MinTime(time_k);
    bm::RegisterBenchmark(name, measure<scalar_at, bytes_per_vector_ak, metric_at>, distance_func)
        ->Threads(threads_k)
        ->MinTime(time_k);
}

simsimd_f32_t cos_f32_naive(simsimd_f32_t* v1, simsimd_f32_t* v2, std::size_t n) {
    simsimd_f32_t inner_product = 0;
    simsimd_f32_t magnitude1 = 0;
    simsimd_f32_t magnitude2 = 0;
    for (std::size_t i = 0; i != n; ++i) {
        inner_product += v1[i] * v2[i];
        magnitude1 += v1[i] * v1[i];
        magnitude2 += v2[i] * v2[i];
    }
    return 1 - inner_product / (std::sqrt(magnitude1) * std::sqrt(magnitude2));
}

int main(int argc, char** argv) {

    bool compiled_with_sve = false;
    bool compiled_with_neon = false;
    bool compiled_with_avx2 = false;
    bool compiled_with_avx512popcnt = false;

#if defined(__ARM_FEATURE_SVE)
    compiled_with_sve = true;
#endif
#if defined(__ARM_NEON)
    compiled_with_neon = true;
#endif
#if defined(__AVX2__)
    compiled_with_avx2 = true;
#endif
#if defined(__AVX512VPOPCNTDQ__)
    compiled_with_avx512popcnt = true;
#endif

    // Log supported functionality
    char const* flags[2] = {"false", "true"};
    std::printf("Benchmarking Similarity Measures\n");
    std::printf("\n");
    std::printf("- Arm NEON support enabled: %s\n", flags[compiled_with_neon]);
    std::printf("- Arm SVE support enabled: %s\n", flags[compiled_with_sve]);
    std::printf("- x86 AVX2 support enabled: %s\n", flags[compiled_with_avx2]);
    std::printf("- x86 AVX512VPOPCNTDQ support enabled: %s\n", flags[compiled_with_avx512popcnt]);
    std::printf("Default vector length: %zu bytes\n", bytes_k);
    std::printf("\n");

    // Run the benchmarks
    bm::Initialize(&argc, argv);
    if (bm::ReportUnrecognizedArguments(argc, argv))
        return 1;

    register_<std::uint8_t, 21>("tanimoto_maccs_naive", simsimd_tanimoto_maccs_naive);
    register_<simsimd_f32_t>("cos_f32_naive", cos_f32_naive);

#if defined(__ARM_FEATURE_SVE)
    register_<simsimd_f32_t>("dot_f32_sve", simsimd_dot_f32_sve);
    register_<simsimd_f32_t>("cos_f32_sve", simsimd_cos_f32_sve);
    register_<std::int16_t>("cos_f16_sve", simsimd_cos_f16_sve);
    register_<simsimd_f32_t>("l2sq_f32_sve", simsimd_l2sq_f32_sve);
    register_<std::int16_t>("l2sq_f16_sve", simsimd_l2sq_f16_sve);
    register_<std::uint8_t>("hamming_b1x8_sve", simsimd_hamming_b1x8_sve);
    register_<std::uint8_t>("hamming_b1x128_sve", simsimd_hamming_b1x128_sve);
    register_<std::uint8_t, 21>("tanimoto_maccs_sve", simsimd_tanimoto_maccs_sve);
#endif

#if defined(__ARM_NEON)
    register_<simsimd_f32_t>("dot_f32x4_neon", simsimd_dot_f32x4_neon);
    register_<std::int16_t>("cos_f16x4_neon", simsimd_cos_f16x4_neon);
    register_<std::int8_t>("cos_i8x16_neon", simsimd_cos_i8x16_neon);
    register_<simsimd_f32_t>("cos_f32x4_neon", simsimd_cos_f32x4_neon);
    register_<std::uint8_t, 21>("tanimoto_maccs_neon", simsimd_tanimoto_maccs_neon);
#endif

#if defined(__AVX2__)
    register_<simsimd_f32_t>("dot_f32x4_avx2", simsimd_dot_f32x4_avx2);
    register_<simsimd_f32_t>("cos_f32x4_avx2", simsimd_cos_f32x4_avx2);
#endif

#if defined(__AVX512F__)
    register_<std::int16_t>("cos_f16x16_avx512", simsimd_cos_f16x16_avx512);
    register_<std::uint8_t>("hamming_b1x128_avx512", simsimd_hamming_b1x128_avx512);
    register_<std::uint8_t, 21>("tanimoto_maccs_avx512", simsimd_tanimoto_maccs_avx512);
#endif



    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}