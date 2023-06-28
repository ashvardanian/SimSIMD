#include <thread>

#include <benchmark/benchmark.h>

#include <simsimd/simsimd.h>

namespace bm = benchmark;

static const std::size_t threads_k = std::thread::hardware_concurrency();
static constexpr std::size_t time_k = 10;
static constexpr std::size_t bytes_k = 256;

template <typename scalar_at, std::size_t bytes_per_vector_ak = bytes_k,
          typename metric_at = void> //
static void measure(bm::State& state, metric_at metric) {

    constexpr ::size_t dimensions_ak = bytes_per_vector_ak / sizeof(scalar_at);
    alignas(64) scalar_at a[dimensions_ak]{};
    alignas(64) scalar_at b[dimensions_ak]{};
    float c{};

    std::fill_n(a, dimensions_ak, static_cast<scalar_at>(1));
    std::fill_n(b, dimensions_ak, static_cast<scalar_at>(2));

    for (auto _ : state)
        bm::DoNotOptimize((c = metric(a, b, dimensions_ak)));

    state.SetBytesProcessed(state.iterations() * bytes_per_vector_ak * 2u);
    state.SetItemsProcessed(state.iterations());
}

template <typename scalar_at, std::size_t bytes_per_vector_ak = bytes_k, typename metric_at = void>
void register_(char const* name, metric_at distance_func) {
    bm::RegisterBenchmark(name, measure<scalar_at, bytes_per_vector_ak, metric_at>, distance_func)
        ->Threads(threads_k)
        ->MinTime(time_k);
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
    std::printf("\n");

    // Run the benchmarks
    bm::Initialize(&argc, argv);
    if (bm::ReportUnrecognizedArguments(argc, argv))
        return 1;

    register_<std::uint8_t, 21>("tanimoto_b1x8_naive", simsimd_tanimoto_b1x8_naive);
    register_<std::uint8_t, 21>("tanimoto_b1x8x21_naive", simsimd_tanimoto_b1x8x21_naive);

#if defined(__ARM_FEATURE_SVE)
    register_<simsimd_f32_t>("dot_f32_sve", simsimd_dot_f32_sve);
    register_<simsimd_f32_t>("cos_f32_sve", simsimd_cos_f32_sve);
    register_<simsimd_f32_t>("l2sq_f32_sve", simsimd_l2sq_f32_sve);
    register_<std::int16_t>("l2sq_f16_sve", simsimd_l2sq_f16_sve);
    register_<std::uint8_t>("hamming_b1x8_sve", simsimd_hamming_b1x8_sve);
    register_<std::uint8_t>("hamming_b1x128_sve", simsimd_hamming_b1x128_sve);
    register_<std::uint8_t, 21>("tanimoto_b1x8x21_avx512", simsimd_tanimoto_b1x8x21_avx512);
#endif

#if defined(__ARM_NEON)
    register_<simsimd_f32_t>("dot_f32x4_neon", simsimd_dot_f32x4_neon);
    register_<std::int16_t>("cos_f16x4_neon", simsimd_cos_f16x4_neon);
    register_<std::int8_t>("cos_i8x16_neon", simsimd_cos_i8x16_neon);
    register_<simsimd_f32_t>("cos_f32x4_neon", simsimd_cos_f32x4_neon);
#endif

#if defined(__AVX2__)
    register_<simsimd_f32_t>("dot_f32x4_avx2", simsimd_dot_f32x4_avx2);
    register_<std::int8_t>("dot_i8x16_avx2", simsimd_dot_i8x16_avx2);
    register_<simsimd_f32_t>("cos_f32x4_avx2", simsimd_cos_f32x4_avx2);
#endif

#if defined(__AVX512F__)
    register_<std::int16_t>("cos_f16x16_avx512", simsimd_cos_f16x16_avx512);
    register_<std::uint8_t>("hamming_b1x128_avx512", simsimd_hamming_b1x128_avx512);
#endif

    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}