#include <thread>

#include <benchmark/benchmark.h>

#include <simsimd/simsimd.h>

namespace bm = benchmark;

static const std::size_t threads_k = 1; // std::thread::hardware_concurrency();
static constexpr std::size_t time_k = 2;
static constexpr std::size_t bytes_k = 1024;

template <typename return_at, typename... args_at>
constexpr std::size_t number_of_arguments(return_at (*f)(args_at...)) {
    return sizeof...(args_at);
}

template <typename scalar_at, std::size_t bytes_per_vector_ak = bytes_k> struct vectors_pair_gt {
    static constexpr std::size_t dimensions_ak = bytes_per_vector_ak / sizeof(scalar_at);
    alignas(64) scalar_at a[dimensions_ak]{};
    alignas(64) scalar_at b[dimensions_ak]{};

    std::size_t dimensions() const noexcept { return dimensions_ak; }
    void randomize() noexcept {
        for (std::size_t i = 0; i != dimensions_ak; ++i) {

            if constexpr (std::is_integral_v<scalar_at>)
                a[i] = static_cast<scalar_at>(rand()), b[i] = static_cast<scalar_at>(rand());
            else
                a[i] = static_cast<scalar_at>(float(rand()) / float(RAND_MAX)),
                b[i] = static_cast<scalar_at>(float(rand()) / float(RAND_MAX));
        }
    }
};

template <typename scalar_at, std::size_t bytes_per_vector_ak = bytes_k, typename metric_at = void>
static void measure(bm::State& state, metric_at metric, metric_at baseline) {

    vectors_pair_gt<scalar_at, bytes_per_vector_ak> pair;
    pair.randomize();

    double c{};
    std::size_t iterations = 0;
    for (auto _ : state)
        if constexpr (number_of_arguments(metric_at{}) == 3)
            bm::DoNotOptimize((c = metric(pair.a, pair.b, pair.dimensions()))), iterations++;
        else
            bm::DoNotOptimize((c = metric(pair.a, pair.b))), iterations++;

    state.counters["bytes"] = bm::Counter(iterations * bytes_per_vector_ak * 2u, bm::Counter::kIsRate);
    state.counters["pairs"] = bm::Counter(iterations, bm::Counter::kIsRate);

    double c_baseline = baseline(pair.a, pair.b, pair.dimensions());
    double delta = std::abs(c - c_baseline);
    if (delta < 0.001)
        delta = 0;
    state.counters["delta"] = delta;
}

template <typename scalar_at, std::size_t bytes_per_vector_ak = bytes_k, typename metric_at = void>
void register_(char const* name, metric_at distance_func, metric_at baseline_func) {
    bm::RegisterBenchmark(name, measure<scalar_at, bytes_per_vector_ak, metric_at>, distance_func, baseline_func)
        ->Threads(1)
        ->MinTime(time_k);
    if (threads_k > 1)
        bm::RegisterBenchmark(name, measure<scalar_at, bytes_per_vector_ak, metric_at>, distance_func, baseline_func)
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
    std::printf("Default vector length: %zu bytes\n", bytes_k);
    std::printf("\n");

    // Run the benchmarks
    bm::Initialize(&argc, argv);
    if (bm::ReportUnrecognizedArguments(argc, argv))
        return 1;

#if SIMSIMD_TARGET_ARM_NEON
    register_<simsimd_f16_t>("neon_f16_ip", simsimd_neon_f16_ip, simsimd_accurate_f16_ip);
    register_<simsimd_f16_t>("neon_f16_cos", simsimd_neon_f16_cos, simsimd_accurate_f16_cos);
    register_<simsimd_f16_t>("neon_f16_l2sq", simsimd_neon_f16_l2sq, simsimd_accurate_f16_l2sq);

    register_<simsimd_f32_t>("neon_f32_ip", simsimd_neon_f32_ip, simsimd_accurate_f32_ip);
    register_<simsimd_f32_t>("neon_f32_cos", simsimd_neon_f32_cos, simsimd_accurate_f32_cos);
    register_<simsimd_f32_t>("neon_f32_l2sq", simsimd_neon_f32_l2sq, simsimd_accurate_f32_l2sq);

    register_<simsimd_i8_t>("neon_i8_cos", simsimd_neon_i8_cos, simsimd_accurate_i8_cos);
    register_<simsimd_i8_t>("neon_i8_l2sq", simsimd_neon_i8_l2sq, simsimd_accurate_i8_l2sq);
#endif

#if SIMSIMD_TARGET_ARM_SVE
    register_<simsimd_f16_t>("sve_f16_ip", simsimd_sve_f16_ip, simsimd_accurate_f16_ip);
    register_<simsimd_f16_t>("sve_f16_cos", simsimd_sve_f16_cos, simsimd_accurate_f16_cos);
    register_<simsimd_f16_t>("sve_f16_l2sq", simsimd_sve_f16_l2sq, simsimd_accurate_f16_l2sq);

    register_<simsimd_f32_t>("sve_f32_ip", simsimd_sve_f32_ip, simsimd_accurate_f32_ip);
    register_<simsimd_f32_t>("sve_f32_cos", simsimd_sve_f32_cos, simsimd_accurate_f32_cos);
    register_<simsimd_f32_t>("sve_f32_l2sq", simsimd_sve_f32_l2sq, simsimd_accurate_f32_l2sq);
#endif

#if SIMSIMD_TARGET_X86_AVX2
    register_<simsimd_f16_t>("avx2_f16_ip", simsimd_avx2_f16_ip, simsimd_accurate_f16_ip);
    register_<simsimd_f16_t>("avx2_f16_cos", simsimd_avx2_f16_cos, simsimd_accurate_f16_cos);
    register_<simsimd_f16_t>("avx2_f16_l2sq", simsimd_avx2_f16_l2sq, simsimd_accurate_f16_l2sq);

    register_<simsimd_i8_t>("avx2_i8_cos", simsimd_avx2_i8_cos, simsimd_accurate_i8_cos);
    register_<simsimd_i8_t>("avx2_i8_l2sq", simsimd_avx2_i8_l2sq, simsimd_accurate_i8_l2sq);
#endif

#if SIMSIMD_TARGET_X86_AVX512
    register_<simsimd_f16_t>("avx512_f16_ip", simsimd_avx512_f16_ip, simsimd_accurate_f16_ip);
    register_<simsimd_f16_t>("avx512_f16_cos", simsimd_avx512_f16_cos, simsimd_accurate_f16_cos);
    register_<simsimd_f16_t>("avx512_f16_l2sq", simsimd_avx512_f16_l2sq, simsimd_accurate_f16_l2sq);
#endif

    register_<simsimd_f16_t>("auto_f16_ip", simsimd_auto_f16_ip, simsimd_accurate_f16_ip);
    register_<simsimd_f16_t>("auto_f16_cos", simsimd_auto_f16_cos, simsimd_accurate_f16_cos);
    register_<simsimd_f16_t>("auto_f16_l2sq", simsimd_auto_f16_l2sq, simsimd_accurate_f16_l2sq);

    register_<simsimd_f32_t>("auto_f32_ip", simsimd_auto_f32_ip, simsimd_accurate_f32_ip);
    register_<simsimd_f32_t>("auto_f32_cos", simsimd_auto_f32_cos, simsimd_accurate_f32_cos);
    register_<simsimd_f32_t>("auto_f32_l2sq", simsimd_auto_f32_l2sq, simsimd_accurate_f32_l2sq);

    register_<simsimd_i8_t>("auto_i8_cos", simsimd_auto_i8_cos, simsimd_accurate_i8_cos);
    register_<simsimd_i8_t>("auto_i8_l2sq", simsimd_auto_i8_l2sq, simsimd_accurate_i8_l2sq);

    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}