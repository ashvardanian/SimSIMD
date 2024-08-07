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
    static constexpr bool is_integral = datatype_ak == simsimd_datatype_i8_k || datatype_ak == simsimd_datatype_b8_k;

    alignas(64) scalar_t a[dimensions_ak]{};
    alignas(64) scalar_t b[dimensions_ak]{};

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
            if constexpr (is_integral)
                a[i] = static_cast<scalar_t>(std::rand() % std::numeric_limits<scalar_t>::max()),
                b[i] = static_cast<scalar_t>(std::rand() % std::numeric_limits<scalar_t>::max());
            else {
                double ai = double(std::rand()) / double(RAND_MAX), bi = double(std::rand()) / double(RAND_MAX);
                a2_sum += ai * ai, b2_sum += bi * bi;
                compress(ai, a[i]), compress(bi, b[i]);
            }
        }

        // Normalize the vectors:
        if constexpr (!is_integral) {
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
    constexpr std::size_t pairs_count = 4;
    std::vector<pair_at> pairs(pairs_count);
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
        bm::DoNotOptimize((call_contender(pairs[iterations & (pairs_count - 1)]))), iterations++;

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

    using pair_dims_t = vectors_pair_gt<datatype_ak, 4096>;
    using scalar_t = typename pair_dims_t::scalar_t;
    using pair_bytes_t = vectors_pair_gt<datatype_ak, 4096 / sizeof(scalar_t)>;

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

namespace AVX512_harley_seal {

uint8_t lookup8bit[256] = {
    /* 0 */ 0,  /* 1 */ 1,  /* 2 */ 1,  /* 3 */ 2,
    /* 4 */ 1,  /* 5 */ 2,  /* 6 */ 2,  /* 7 */ 3,
    /* 8 */ 1,  /* 9 */ 2,  /* a */ 2,  /* b */ 3,
    /* c */ 2,  /* d */ 3,  /* e */ 3,  /* f */ 4,
    /* 10 */ 1, /* 11 */ 2, /* 12 */ 2, /* 13 */ 3,
    /* 14 */ 2, /* 15 */ 3, /* 16 */ 3, /* 17 */ 4,
    /* 18 */ 2, /* 19 */ 3, /* 1a */ 3, /* 1b */ 4,
    /* 1c */ 3, /* 1d */ 4, /* 1e */ 4, /* 1f */ 5,
    /* 20 */ 1, /* 21 */ 2, /* 22 */ 2, /* 23 */ 3,
    /* 24 */ 2, /* 25 */ 3, /* 26 */ 3, /* 27 */ 4,
    /* 28 */ 2, /* 29 */ 3, /* 2a */ 3, /* 2b */ 4,
    /* 2c */ 3, /* 2d */ 4, /* 2e */ 4, /* 2f */ 5,
    /* 30 */ 2, /* 31 */ 3, /* 32 */ 3, /* 33 */ 4,
    /* 34 */ 3, /* 35 */ 4, /* 36 */ 4, /* 37 */ 5,
    /* 38 */ 3, /* 39 */ 4, /* 3a */ 4, /* 3b */ 5,
    /* 3c */ 4, /* 3d */ 5, /* 3e */ 5, /* 3f */ 6,
    /* 40 */ 1, /* 41 */ 2, /* 42 */ 2, /* 43 */ 3,
    /* 44 */ 2, /* 45 */ 3, /* 46 */ 3, /* 47 */ 4,
    /* 48 */ 2, /* 49 */ 3, /* 4a */ 3, /* 4b */ 4,
    /* 4c */ 3, /* 4d */ 4, /* 4e */ 4, /* 4f */ 5,
    /* 50 */ 2, /* 51 */ 3, /* 52 */ 3, /* 53 */ 4,
    /* 54 */ 3, /* 55 */ 4, /* 56 */ 4, /* 57 */ 5,
    /* 58 */ 3, /* 59 */ 4, /* 5a */ 4, /* 5b */ 5,
    /* 5c */ 4, /* 5d */ 5, /* 5e */ 5, /* 5f */ 6,
    /* 60 */ 2, /* 61 */ 3, /* 62 */ 3, /* 63 */ 4,
    /* 64 */ 3, /* 65 */ 4, /* 66 */ 4, /* 67 */ 5,
    /* 68 */ 3, /* 69 */ 4, /* 6a */ 4, /* 6b */ 5,
    /* 6c */ 4, /* 6d */ 5, /* 6e */ 5, /* 6f */ 6,
    /* 70 */ 3, /* 71 */ 4, /* 72 */ 4, /* 73 */ 5,
    /* 74 */ 4, /* 75 */ 5, /* 76 */ 5, /* 77 */ 6,
    /* 78 */ 4, /* 79 */ 5, /* 7a */ 5, /* 7b */ 6,
    /* 7c */ 5, /* 7d */ 6, /* 7e */ 6, /* 7f */ 7,
    /* 80 */ 1, /* 81 */ 2, /* 82 */ 2, /* 83 */ 3,
    /* 84 */ 2, /* 85 */ 3, /* 86 */ 3, /* 87 */ 4,
    /* 88 */ 2, /* 89 */ 3, /* 8a */ 3, /* 8b */ 4,
    /* 8c */ 3, /* 8d */ 4, /* 8e */ 4, /* 8f */ 5,
    /* 90 */ 2, /* 91 */ 3, /* 92 */ 3, /* 93 */ 4,
    /* 94 */ 3, /* 95 */ 4, /* 96 */ 4, /* 97 */ 5,
    /* 98 */ 3, /* 99 */ 4, /* 9a */ 4, /* 9b */ 5,
    /* 9c */ 4, /* 9d */ 5, /* 9e */ 5, /* 9f */ 6,
    /* a0 */ 2, /* a1 */ 3, /* a2 */ 3, /* a3 */ 4,
    /* a4 */ 3, /* a5 */ 4, /* a6 */ 4, /* a7 */ 5,
    /* a8 */ 3, /* a9 */ 4, /* aa */ 4, /* ab */ 5,
    /* ac */ 4, /* ad */ 5, /* ae */ 5, /* af */ 6,
    /* b0 */ 3, /* b1 */ 4, /* b2 */ 4, /* b3 */ 5,
    /* b4 */ 4, /* b5 */ 5, /* b6 */ 5, /* b7 */ 6,
    /* b8 */ 4, /* b9 */ 5, /* ba */ 5, /* bb */ 6,
    /* bc */ 5, /* bd */ 6, /* be */ 6, /* bf */ 7,
    /* c0 */ 2, /* c1 */ 3, /* c2 */ 3, /* c3 */ 4,
    /* c4 */ 3, /* c5 */ 4, /* c6 */ 4, /* c7 */ 5,
    /* c8 */ 3, /* c9 */ 4, /* ca */ 4, /* cb */ 5,
    /* cc */ 4, /* cd */ 5, /* ce */ 5, /* cf */ 6,
    /* d0 */ 3, /* d1 */ 4, /* d2 */ 4, /* d3 */ 5,
    /* d4 */ 4, /* d5 */ 5, /* d6 */ 5, /* d7 */ 6,
    /* d8 */ 4, /* d9 */ 5, /* da */ 5, /* db */ 6,
    /* dc */ 5, /* dd */ 6, /* de */ 6, /* df */ 7,
    /* e0 */ 3, /* e1 */ 4, /* e2 */ 4, /* e3 */ 5,
    /* e4 */ 4, /* e5 */ 5, /* e6 */ 5, /* e7 */ 6,
    /* e8 */ 4, /* e9 */ 5, /* ea */ 5, /* eb */ 6,
    /* ec */ 5, /* ed */ 6, /* ee */ 6, /* ef */ 7,
    /* f0 */ 4, /* f1 */ 5, /* f2 */ 5, /* f3 */ 6,
    /* f4 */ 5, /* f5 */ 6, /* f6 */ 6, /* f7 */ 7,
    /* f8 */ 5, /* f9 */ 6, /* fa */ 6, /* fb */ 7,
    /* fc */ 6, /* fd */ 7, /* fe */ 7, /* ff */ 8};

uint64_t lower_qword(const __m128i v) { return _mm_cvtsi128_si64(v); }

uint64_t higher_qword(const __m128i v) { return lower_qword(_mm_srli_si128(v, 8)); }

uint64_t simd_sum_epu64(const __m128i v) { return lower_qword(v) + higher_qword(v); }

uint64_t simd_sum_epu64(const __m256i v) {

    return static_cast<uint64_t>(_mm256_extract_epi64(v, 0)) + static_cast<uint64_t>(_mm256_extract_epi64(v, 1)) +
           static_cast<uint64_t>(_mm256_extract_epi64(v, 2)) + static_cast<uint64_t>(_mm256_extract_epi64(v, 3));
}

uint64_t simd_sum_epu64(const __m512i v) {

    const __m256i lo = _mm512_extracti64x4_epi64(v, 0);
    const __m256i hi = _mm512_extracti64x4_epi64(v, 1);

    return simd_sum_epu64(lo) + simd_sum_epu64(hi);
}

__m512i popcount(const __m512i v) {
    const __m512i m1 = _mm512_set1_epi8(0x55);
    const __m512i m2 = _mm512_set1_epi8(0x33);
    const __m512i m4 = _mm512_set1_epi8(0x0F);

    const __m512i t1 = _mm512_sub_epi8(v, (_mm512_srli_epi16(v, 1) & m1));
    const __m512i t2 = _mm512_add_epi8(t1 & m2, (_mm512_srli_epi16(t1, 2) & m2));
    const __m512i t3 = _mm512_add_epi8(t2, _mm512_srli_epi16(t2, 4)) & m4;
    return _mm512_sad_epu8(t3, _mm512_setzero_si512());
}

void CSA(__m512i& h, __m512i& l, __m512i a, __m512i b, __m512i c) {
    /*
        c b a | l h
        ------+----
        0 0 0 | 0 0
        0 0 1 | 1 0
        0 1 0 | 1 0
        0 1 1 | 0 1
        1 0 0 | 1 0
        1 0 1 | 0 1
        1 1 0 | 0 1
        1 1 1 | 1 1

        l - digit
        h - carry
    */

    l = _mm512_ternarylogic_epi32(c, b, a, 0x96);
    h = _mm512_ternarylogic_epi32(c, b, a, 0xe8);
}

uint64_t popcnt(__m512i const* a, __m512i const* b, const uint64_t size) {
    __m512i total = _mm512_setzero_si512();
    __m512i ones = _mm512_setzero_si512();
    __m512i twos = _mm512_setzero_si512();
    __m512i fours = _mm512_setzero_si512();
    __m512i eights = _mm512_setzero_si512();
    __m512i sixteens = _mm512_setzero_si512();
    __m512i twosA, twosB, foursA, foursB, eightsA, eightsB;

    const uint64_t limit = size - size % 16;
    uint64_t i = 0;

    for (; i < limit; i += 16) {
        CSA(twosA, ones, ones, a[i + 0] ^ b[i + 0], a[i + 1] ^ b[i + 1]);
        CSA(twosB, ones, ones, a[i + 2] ^ b[i + 2], a[i + 3] ^ b[i + 3]);
        CSA(foursA, twos, twos, twosA, twosB);
        CSA(twosA, ones, ones, a[i + 4] ^ b[i + 4], a[i + 5] ^ b[i + 5]);
        CSA(twosB, ones, ones, a[i + 6] ^ b[i + 6], a[i + 7] ^ b[i + 7]);
        CSA(foursB, twos, twos, twosA, twosB);
        CSA(eightsA, fours, fours, foursA, foursB);
        CSA(twosA, ones, ones, a[i + 8] ^ b[i + 8], a[i + 9] ^ b[i + 9]);
        CSA(twosB, ones, ones, a[i + 10] ^ b[i + 10], a[i + 11] ^ b[i + 11]);
        CSA(foursA, twos, twos, twosA, twosB);
        CSA(twosA, ones, ones, a[i + 12] ^ b[i + 12], a[i + 13] ^ b[i + 13]);
        CSA(twosB, ones, ones, a[i + 14] ^ b[i + 14], a[i + 15] ^ b[i + 15]);
        CSA(foursB, twos, twos, twosA, twosB);
        CSA(eightsB, fours, fours, foursA, foursB);
        CSA(sixteens, eights, eights, eightsA, eightsB);

        total = _mm512_add_epi64(total, popcount(sixteens));
    }

    total = _mm512_slli_epi64(total, 4);                                     // * 16
    total = _mm512_add_epi64(total, _mm512_slli_epi64(popcount(eights), 3)); // += 8 * ...
    total = _mm512_add_epi64(total, _mm512_slli_epi64(popcount(fours), 2));  // += 4 * ...
    total = _mm512_add_epi64(total, _mm512_slli_epi64(popcount(twos), 1));   // += 2 * ...
    total = _mm512_add_epi64(total, popcount(ones));

    for (; i < size; i++)
        total = _mm512_add_epi64(total, popcount(a[i] ^ b[i]));

    return simd_sum_epu64(total);
}

} // namespace AVX512_harley_seal

void popcnt_AVX512_harley_seal(simsimd_b8_t const* a, simsimd_b8_t const* b, simsimd_size_t size,
                               simsimd_distance_t* results) {
    uint64_t total = AVX512_harley_seal::popcnt((const __m512i*)a, (const __m512i*)b, size / 64);

    for (size_t i = size - size % 64; i < size; i++)
        total += AVX512_harley_seal::lookup8bit[a[i] ^ b[i]];

    results[0] = total;
}

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
    std::printf("- Arm NEON F16 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_neon_f16_k) != 0]);
    std::printf("- Arm NEON BF16 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_neon_bf16_k) != 0]);
    std::printf("- Arm NEON I8 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_neon_i8_k) != 0]);
    std::printf("- Arm SVE support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sve_k) != 0]);
    std::printf("- Arm SVE F16 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sve_f16_k) != 0]);
    std::printf("- Arm SVE BF16 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sve_bf16_k) != 0]);
    std::printf("- Arm SVE I8 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sve_i8_k) != 0]);
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

    register_<simsimd_datatype_bf16_k>("dot_bf16_neon", simsimd_dot_bf16_neon, simsimd_dot_bf16_accurate);
    register_<simsimd_datatype_bf16_k>("cos_bf16_neon", simsimd_cos_bf16_neon, simsimd_cos_bf16_accurate);
    register_<simsimd_datatype_bf16_k>("l2sq_bf16_neon", simsimd_l2sq_bf16_neon, simsimd_l2sq_bf16_accurate);

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

    register_<simsimd_datatype_bf16c_k>("dot_bf16c_neon", simsimd_dot_bf16c_neon, simsimd_dot_bf16c_accurate);
    register_<simsimd_datatype_bf16c_k>("vdot_bf16c_neon", simsimd_vdot_bf16c_neon, simsimd_vdot_bf16c_accurate);
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
    register_<simsimd_datatype_b8_k>("hamming_b8_icehs", popcnt_AVX512_harley_seal, simsimd_hamming_b8_serial);
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
