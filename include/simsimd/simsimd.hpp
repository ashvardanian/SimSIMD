/**
 *  @brief Collection of Similarity Measures, SIMD-accelerated with SSE, AVX, NEON, SVE.
 *
 *  @author Ashot Vardanian
 *  @date March 14, 2023
 *
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  Detecting target CPU features at compile time: https://stackoverflow.com/a/28939692/2766161
 */

#pragma once
#include <cmath> // `std::sqrt`

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

namespace av {
namespace simsimd {

using dim_t = unsigned int;
using i8_t = signed char;
using u8_t = unsigned char;
using i16_t = short;
using i32_t = int;
using u32_t = unsigned int;
using u64_t = unsigned long;
using f32_t = float;
using f64_t = double;
enum class f16_t : i16_t {};

union f32i32_t {
    i32_t i;
    f32_t f;
};

/**
 *  @brief Dot product for arbitrary length/type vectors.
 *  @return Dot product ∈ [-1, 1] for normalized vectors.
 */
struct dot_product_t {

    f32_t operator()(f32_t const* a, f32_t const* b, dim_t const dim) const noexcept {
#if defined(__ARM_FEATURE_SVE)
        dim_t i = 0;
        svfloat32_t ab_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
        svbool_t pg_vec = svwhilelt_b32(i, dim);
        do {
            svfloat32_t a_vec = svld1_f32(pg_vec, a + i);
            svfloat32_t b_vec = svld1_f32(pg_vec, b + i);
            ab_vec = svmla_f32_x(pg_vec, ab_vec, a_vec, b_vec);
            i += svcntw();
            pg_vec = svwhilelt_b32(i, dim);
        } while (svptest_any(svptrue_b32(), pg_vec));
        return svaddv_f32(svptrue_b32(), ab_vec);
#else
        return any(a, b, dim);
#endif
    }

    template <typename at> //
    at operator()(at const* a, at const* b, dim_t const dim) const noexcept {
        return any(a, b, dim);
    }

  private:
    template <typename at> //
    inline at any(at const* a, at const* b, dim_t const dim) const noexcept {
        at ab = 0;
#pragma GCC ivdep
#pragma clang loop vectorize(enable)
        for (dim_t i = 0; i != dim; ++i)
            ab += a[i] * b[i];
        return ab;
    }
};

/**
 *  @brief Cosine similarity for arbitrary length/type vectors.
 *  @return Similarity ∈ [-1, 1].
 */
struct cosine_similarity_t {

    f32_t operator()(f32_t const* a, f32_t const* b, dim_t const dim) const noexcept {
#if defined(__ARM_FEATURE_SVE)
        dim_t i = 0;
        svfloat32_t ab_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
        svfloat32_t a2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
        svfloat32_t b2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
        svbool_t pg_vec = svwhilelt_b32(i, dim);
        do {
            svfloat32_t a_vec = svld1_f32(pg_vec, a + i);
            svfloat32_t b_vec = svld1_f32(pg_vec, b + i);
            ab_vec = svmla_f32_x(pg_vec, ab_vec, a_vec, b_vec);
            a2_vec = svmla_f32_x(pg_vec, a2_vec, a_vec, a_vec);
            b2_vec = svmla_f32_x(pg_vec, b2_vec, b_vec, b_vec);
            i += svcntw();
            pg_vec = svwhilelt_b32(i, dim);
        } while (svptest_any(svptrue_b32(), pg_vec));
        auto ab = svaddv_f32(svptrue_b32(), ab_vec);
        auto a2 = svaddv_f32(svptrue_b32(), a2_vec);
        auto b2 = svaddv_f32(svptrue_b32(), b2_vec);
        return ab / (std::sqrt(a2) * std::sqrt(b2));
#else
        return any(a, b, dim);
#endif
    }

    template <typename at> //
    at operator()(at const* a, at const* b, dim_t const dim) const noexcept {
        return any(a, b, dim);
    }

  private:
    template <typename at> //
    inline at any(at const* a, at const* b, dim_t const dim) const noexcept {
        at ab = 0, a2 = 0, b2 = 0;
#pragma GCC ivdep
#pragma clang loop vectorize(enable)
        for (dim_t i = 0; i != dim; ++i)
            ab += a[i] * b[i], a2 += a[i] * a[i], b2 += b[i] * b[i];
        return ab / (std::sqrt(a2) * std::sqrt(b2));
    }
};

/**
 *  @brief Euclidean distance (L2) for arbitrary length/type vectors.
 *  @return Euclidean distance (L2) ∈ [0, 2] for normalized vectors.
 */
struct euclidean_distance_t {

    f32_t operator()(i8_t const* a, i8_t const* b, dim_t const dim) const noexcept {
        i32_t d2 = 0;
#pragma GCC ivdep
#pragma clang loop vectorize(enable)
        for (dim_t i = 0; i != dim; ++i)
            d2 += i32_t(i16_t(a[i]) - i16_t(b[i])) * i32_t(i16_t(a[i]) - i16_t(b[i]));
        return std::sqrt(d2);
    }
    template <typename at> //
    at operator()(at const* a, at const* b, dim_t const dim) const noexcept {
        at d2 = 0;
#pragma GCC ivdep
#pragma clang loop vectorize(enable)
        for (dim_t i = 0; i != dim; ++i)
            d2 += (a[i] - b[i]) * (a[i] - b[i]);
        return std::sqrt(d2);
    }
};

/**
 *  @brief Bitwise Hamming distance on scalar words (8, 16, 32, or 64-bit).
 *  @return Integer number of different bits ∈ [0, dim).
 */
struct hamming_bits_distance_t {

#if defined(__GNUC__)
    static dim_t popcount(u32_t v) noexcept { return __builtin_popcount(v); }
    static dim_t popcount(u64_t v) noexcept { return __builtin_popcountl(v); }
#elif defined(__AVX2__)
    static dim_t popcount(u32_t v) noexcept { return _mm_popcnt_u32(v); }
    static dim_t popcount(u64_t v) noexcept { return _mm_popcnt_u64(v); }
#endif

    template <typename at> //
    dim_t operator()(at const* a, at const* b, dim_t const dim) const noexcept {
        dim_t words = dim / (sizeof(at) * __CHAR_BIT__);
#if defined(__ARM_FEATURE_SVE)
        auto a8 = reinterpret_cast<u8_t const*>(a);
        auto b8 = reinterpret_cast<u8_t const*>(b);
        dim_t i = 0;
        svuint8_t d_vec = svdupq_n_u8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        svbool_t pg_vec = svwhilelt_b8(i, dim);
        do {
            svuint8_t a_vec = svld1_u8(pg_vec, a8 + i);
            svuint8_t b_vec = svld1_u8(pg_vec, b8 + i);
            svuint8_t a_xor_b_vec = sveor_u8_m(pg_vec, a_vec, b_vec);
            d_vec = svadd_u8_m(pg_vec, d_vec, svcnt_u8_x(pg_vec, a_xor_b_vec));
            i += svcntb();
            pg_vec = svwhilelt_b32(i, dim);
        } while (svptest_any(svptrue_b32(), pg_vec));
        auto d = svaddv_u8(svptrue_b32(), d_vec);
        return d;
#else
        dim_t d = 0;
#pragma GCC ivdep
#pragma clang loop vectorize(enable)
        for (dim_t i = 0; i != words; ++i)
            d += hamming_bits_distance_t::popcount(a[i] ^ b[i]);
#endif
        return d;
    }
};

/**
 *  @brief Dot product between `float` vectors where `dim` is a multiple of 4.
 *  @return Dot product ∈ [-1, 1] for normalized vectors.
 */
struct dot_product_f32x4k_t {

    f32_t operator()(f32_t const* a, f32_t const* b, dim_t const dim) const noexcept {
#if defined(__AVX2__)
        __m128 ab_vec = _mm_set1_ps(0);
        for (dim_t i = 0; i != dim; i += 4)
            ab_vec = _mm_fmadd_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i), ab_vec);
        ab_vec = _mm_hadd_ps(ab_vec, ab_vec);
        ab_vec = _mm_hadd_ps(ab_vec, ab_vec);
        f32i32_t ab_union = {_mm_cvtsi128_si32(_mm_castps_si128(ab_vec))};
        return ab_union.f;
#elif defined(__ARM_NEON)
        float32x4_t ab_vec = vdupq_n_f32(0);
        for (dim_t i = 0; i != dim; i += 4)
            ab_vec = vmlaq_f32(ab_vec, vld1q_f32(a + i), vld1q_f32(b + i));
        return vaddvq_f32(ab_vec);
#else
        return dot_product_t{}(a, b, dim);
#endif
    }
};

/**
 *  @brief Cosine similarity between `float` vectors where `dim` is a multiple of 4.
 *  @return Similarity ∈ [-1, 1].
 */
struct cosine_similarity_f32x4k_t {

    f32_t operator()(f32_t const* a, f32_t const* b, dim_t const dim) const noexcept {
#if defined(__AVX2__)
        __m128 ab_vec = _mm_set1_ps(0);
        __m128 a2_vec = _mm_set1_ps(0);
        __m128 b2_vec = _mm_set1_ps(0);
        for (dim_t i = 0; i != dim; i += 4) {
            auto a_vec = _mm_loadu_ps(a + i);
            auto b_vec = _mm_loadu_ps(b + i);
            ab_vec = _mm_fmadd_ps(a_vec, b_vec, ab_vec);
            a2_vec = _mm_fmadd_ps(a_vec, a_vec, a2_vec);
            b2_vec = _mm_fmadd_ps(b_vec, b_vec, b2_vec);
        }
        ab_vec = _mm_hadd_ps(ab_vec, ab_vec);
        ab_vec = _mm_hadd_ps(ab_vec, ab_vec);
        a2_vec = _mm_hadd_ps(a2_vec, a2_vec);
        a2_vec = _mm_hadd_ps(a2_vec, a2_vec);
        b2_vec = _mm_hadd_ps(b2_vec, b2_vec);
        b2_vec = _mm_hadd_ps(b2_vec, b2_vec);
        f32i32_t ab_union = {_mm_cvtsi128_si32(_mm_castps_si128(ab_vec))};
        f32i32_t a2_union = {_mm_cvtsi128_si32(_mm_castps_si128(a2_vec))};
        f32i32_t b2_union = {_mm_cvtsi128_si32(_mm_castps_si128(b2_vec))};
        return ab_union.f / (std::sqrt(a2_union.f) * std::sqrt(b2_union.f));
#elif defined(__ARM_NEON)
        float32x4_t ab_vec = vdupq_n_f32(0);
        float32x4_t a2_vec = vdupq_n_f32(0);
        float32x4_t b2_vec = vdupq_n_f32(0);
        for (dim_t i = 0; i != dim; i += 4) {
            auto a_vec = vld1q_f32(a + i);
            auto b_vec = vld1q_f32(b + i);
            ab_vec = vmlaq_f32(ab_vec, a_vec, b_vec);
            a2_vec = vmlaq_f32(a2_vec, a_vec, a_vec);
            b2_vec = vmlaq_f32(b2_vec, b_vec, b_vec);
        }
        auto ab = vaddvq_f32(ab_vec);
        auto a2 = vaddvq_f32(a2_vec);
        auto b2 = vaddvq_f32(b2_vec);
        return ab / (std::sqrt(a2) * std::sqrt(b2));
#else
        return cosine_similarity_t{}(a, b, dim);
#endif
    }
};

/**
 *  @brief Dot product on vectors quantized into the [-100, 100] interval.
 */
struct dot_product_i8x16k_t {

    i32_t operator()(i8_t const* a, i8_t const* b, dim_t const dim) const noexcept {
#if defined(__AVX2__)
        __m256i ab_vec = _mm256_set1_epi16(0);
        for (dim_t i = 0; i != dim; i += 4)
            ab_vec = _mm256_add_epi16(                                              //
                ab_vec,                                                             //
                _mm256_mullo_epi16(                                                 //
                    _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i const*)(a + i))), //
                    _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i const*)(b + i)))));
        ab_vec = _mm256_hadd_epi16(ab_vec, ab_vec);
        ab_vec = _mm256_hadd_epi16(ab_vec, ab_vec);
        ab_vec = _mm256_hadd_epi16(ab_vec, ab_vec);
        return (_mm256_cvtsi256_si32(ab_vec) & 0xFF);
#else
        return 0;
#endif
    }
};

/**
 *  @brief Bitwise Hamming distance on 128-bit long words.
 *  @return Integer number of different bits ∈ [0, dim).
 */
struct hamming_bits_distance_u1x128k_t {

    template <typename at> //
    dim_t operator()(at const* a, at const* b, dim_t const dim) const noexcept {
        auto words = dim / 128;

#if defined(__AVX512VPOPCNTDQ__)
        auto a64 = reinterpret_cast<u64_t const*>(a);
        auto b64 = reinterpret_cast<u64_t const*>(b);
        /// Contains 2x 64-bit integers with running population count sums.
        __m128i d_vec = _mm_set_epi64x(0, 0);
        for (dim_t i = 0; i != words; i += 2)
            d_vec = _mm_add_epi64( //
                d_vec,             //
                _mm_popcnt_epi64(  //
                    _mm_xor_si128( //
                        _mm_load_si128(reinterpret_cast<__m128i const*>(a64 + i)),
                        _mm_load_si128(reinterpret_cast<__m128i const*>(b64 + i)))));
        dim_t d = _mm_cvtm64_si64(_mm_movepi64_pi64(d_vec)) + _mm_extract_epi64(d_vec, 1);
        return d;
#elif defined(__ARM_NEON)
        auto a8 = reinterpret_cast<u8_t const*>(a);
        auto b8 = reinterpret_cast<u8_t const*>(b);
        /// Contains 16x 8-bit integers with running population count sums.
        uint8x16_t d_vec = vdupq_n_u8(0);
        for (dim_t i = 0; i != dim; i += 16) {
            auto a_vec = vld1q_u8(a8 + i);
            auto b_vec = vld1q_u8(b8 + i);
            auto a_xor_b_vec = veorq_u8(a_vec, b_vec);
            d_vec = vaddq_u8(d_vec, vcntq_u8(a_xor_b_vec));
        }
        return vaddvq_u8(d_vec);
#else
        auto a64 = reinterpret_cast<u64_t const*>(a);
        auto b64 = reinterpret_cast<u64_t const*>(b);
        dim_t d_odd = 0, d_even = 0;
        for (dim_t i = 0; i != words; i += 2)
            d_even += hamming_bits_distance_t::popcount(a64[i] ^ b64[i]),
                d_odd += hamming_bits_distance_t::popcount(a64[i + 1] ^ b64[i + 1]);
        return d_odd + d_even;
#endif
    }
};

template <typename similarity_measure_at> //
struct cosine_distance_from_similarity_gt {
    template <typename at> //
    inline at operator()(at const* a, at const* b, dim_t const dim) const noexcept {
        return 1 - similarity_measure_at{}(a, b, dim);
    }
};

using cosine_distance_t = cosine_distance_from_similarity_gt<cosine_similarity_t>;
using cosine_distance_f32x4k_t = cosine_distance_from_similarity_gt<cosine_similarity_f32x4k_t>;

template <typename distance_measure_at> //
struct hamming_bits_similarity_from_distance_gt {
    template <typename at> //
    inline f32_t operator()(at const* a, at const* b, dim_t const dim) const noexcept {
        return 1.f - distance_measure_at{}(a, b, dim) * 1.f / dim;
    }
};

using hamming_bits_similarity_t = hamming_bits_similarity_from_distance_gt<hamming_bits_distance_t>;
using hamming_bits_similarity_u1x128k_t = hamming_bits_similarity_from_distance_gt<hamming_bits_distance_u1x128k_t>;

} // namespace simsimd
} // namespace av