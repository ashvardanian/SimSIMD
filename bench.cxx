#include <benchmark/benchmark.h>

#include <simsimd/simsimd.hpp>

namespace bm = benchmark;
using namespace av::simsimd;

static constexpr std::size_t threads_k = 64;
static constexpr std::size_t time_k = 100;

template <typename metric_at, typename scalar_at, std::size_t bytes_per_vector_ak,
          std::size_t dimensions_ak = bytes_per_vector_ak / sizeof(scalar_at)> //
static void measure(bm::State& state) {

    constexpr std::size_t buffer_size_k = bytes_per_vector_ak / sizeof(scalar_at);
    alignas(64) scalar_at a[buffer_size_k]{};
    alignas(64) scalar_at b[buffer_size_k]{};
    scalar_at c{};

    std::fill_n(a, buffer_size_k, 1);
    std::fill_n(b, buffer_size_k, 2);

    for (auto _ : state)
        bm::DoNotOptimize((c = metric_at{}(a, b, dimensions_ak)));

    state.SetBytesProcessed(state.iterations() * bytes_per_vector_ak * 2u);
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE(measure, cosine_similarity_t, f32_t, 32)->Threads(threads_k)->MinTime(time_k);
BENCHMARK_TEMPLATE(measure, cosine_similarity_t, f32_t, 256)->Threads(threads_k)->MinTime(time_k);
BENCHMARK_TEMPLATE(measure, cosine_similarity_f32x4k_t, f32_t, 32)->Threads(threads_k)->MinTime(time_k);
BENCHMARK_TEMPLATE(measure, cosine_similarity_f32x4k_t, f32_t, 256)->Threads(threads_k)->MinTime(time_k);

BENCHMARK_TEMPLATE(measure, hamming_bits_distance_t, u64_t, 32, 256)->Threads(threads_k)->MinTime(time_k);
BENCHMARK_TEMPLATE(measure, hamming_bits_distance_t, u64_t, 256, 2048)->Threads(threads_k)->MinTime(time_k);
BENCHMARK_TEMPLATE(measure, hamming_bits_distance_u1x128k_t, u64_t, 32, 256)->Threads(threads_k)->MinTime(time_k);
BENCHMARK_TEMPLATE(measure, hamming_bits_distance_u1x128k_t, u64_t, 256, 2048)->Threads(threads_k)->MinTime(time_k);

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
    compiled_withavx2 = true;
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
    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}