#include <cmath>  // `std::sqrt`
#include <thread> // `std::thread`

#include <benchmark/benchmark.h>

#define SIMSIMD_RSQRT simsimd_approximate_inverse_square_root
#include <simsimd/simsimd.h>

namespace bm = benchmark;

template <typename return_at, typename... args_at>
constexpr std::size_t number_of_arguments(return_at (*f)(args_at...)) {
    return sizeof...(args_at);
}

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
                a[i] /= a2_sum, b[i] /= b2_sum;
        }
    }
};

template <typename pair_at, typename metric_at = void>
static void measure(bm::State& state, metric_at metric, metric_at baseline) {

    pair_at pair;
    pair.randomize();
    // pair.set(1);

    double c_baseline = baseline(pair.a, pair.b, pair.dimensions());
    double c = 0;
    std::size_t iterations = 0;
    for (auto _ : state)
        bm::DoNotOptimize((c = metric(pair.a, pair.b, pair.dimensions()))), iterations++;

    state.counters["bytes"] = bm::Counter(iterations * pair.size_bytes() * 2, bm::Counter::kIsRate);
    state.counters["pairs"] = bm::Counter(iterations, bm::Counter::kIsRate);

    double delta = std::abs(c - c_baseline) > 0.0001 ? std::abs(c - c_baseline) : 0;
    double error = delta != 0 && c_baseline != 0 ? delta / c_baseline : 0;
    state.counters["abs_delta"] = delta;
    state.counters["relative_error"] = error;
}

template <typename scalar_at, typename metric_at = void>
void register_(std::string name, metric_at* distance_func, metric_at* baseline_func) {

    using pair_dims_t = vectors_pair_gt<scalar_at, 1536>;
    using pair_bytes_t = vectors_pair_gt<scalar_at, 1536 / sizeof(scalar_at)>;

    std::size_t seconds = 10;
    std::size_t threads = std::thread::hardware_concurrency(); // 1;
    std::string name_dims = name + "_" + std::to_string(pair_dims_t{}.dimensions()) + "d";
    std::string name_bytes = name + "_" + std::to_string(pair_bytes_t{}.size_bytes()) + "b";

    bm::RegisterBenchmark(name_dims.c_str(), measure<pair_dims_t, metric_at*>, distance_func, baseline_func)
        ->MinTime(seconds)
        ->Threads(threads);
    bm::RegisterBenchmark(name_bytes.c_str(), measure<pair_bytes_t, metric_at*>, distance_func, baseline_func)
        ->MinTime(seconds)
        ->Threads(threads);
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

    register_<simsimd_i8_t>("avx512_i8_cos", simsimd_avx512_i8_cos, simsimd_accurate_i8_cos);
    register_<simsimd_i8_t>("avx512_i8_l2sq", simsimd_avx512_i8_l2sq, simsimd_accurate_i8_l2sq);

    register_<simsimd_f32_t>("avx512_f32_ip", simsimd_avx512_f32_ip, simsimd_accurate_f32_ip);
    register_<simsimd_f32_t>("avx512_f32_cos", simsimd_avx512_f32_cos, simsimd_accurate_f32_cos);
    register_<simsimd_f32_t>("avx512_f32_l2sq", simsimd_avx512_f32_l2sq, simsimd_accurate_f32_l2sq);
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