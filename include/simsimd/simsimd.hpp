/**
 * @brief Collection of Similarity Measures, SIMD-accelerated with SSE, AVX, NEON, SVE.
 *
 * @author Ashot Vardanian
 * @date March 14, 2023
 *
 * x86: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 * Arm: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 */

#pragma once
#include <cmath> // `std::sqrt`

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

namespace av {
namespace simsimd {

using dim_t = unsigned int;
using i8_t = signed char;
using i16_t = short;
using i32_t = int;
using f32_t = float;
using f64_t = double;
enum class f16_t : i16_t {};

union f32i32_t {
    i32_t i;
    f32_t f;
};

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
        return 1 - ab;
    }
};

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
        return 1 - ab / (std::sqrt(a2) * std::sqrt(b2));
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
        return 1 - ab / (std::sqrt(a2) * std::sqrt(b2));
    }
};

struct euclidean_distance_t {

    f32_t operator()(i8_t const* a, i8_t const* b, dim_t const dim) const noexcept {
        i32_t d2 = 0;
#pragma GCC ivdep
#pragma clang loop vectorize(enable)
        for (dim_t i = 0; i != dim; ++i)
            d2 += i32_t(i16_t(a[i]) - i16_t(b[i])) * i32_t(i16_t(a[i]) - i16_t(b[i]));
        return std::sqrt(d2);
    }
    template <typename at> at operator()(at const* a, at const* b, dim_t const dim) const noexcept {
        at d2 = 0;
#pragma GCC ivdep
#pragma clang loop vectorize(enable)
        for (dim_t i = 0; i != dim; ++i)
            d2 += (a[i] - b[i]) * (a[i] - b[i]);
        return std::sqrt(d2);
    }
};

/**
 * @brief
 *      SIMD-accelerated dot-product distance, that assumes vector sizes to be multiples
 *      of 128 bits, and address of the first argument to be aligned to the same size registers.
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
        return 1 - ab_union.f;
#elif defined(__ARM_NEON)
        float32x4_t ab_vec = vdupq_n_f32(0);
        for (dim_t i = 0; i != dim; i += 4)
            ab_vec = vmlaq_f32(ab_vec, vld1q_f32(a + i), vld1q_f32(b + i));
        return 1 - vaddvq_f32(ab_vec);
#else
        return dot_product_t{}(a, b, dim);
#endif
    }
};

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
        return 1 - ab_union.f / (std::sqrt(a2_union.f) * std::sqrt(b2_union.f));
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
        return 1 - ab / (std::sqrt(a2) * std::sqrt(b2));
#else
        return cosine_similarity_t{}(a, b, dim);
#endif
    }
};

/** @brief Takes scalars quantized into a [-100; 100] interval. */
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
        return 1 - (_mm256_cvtsi256_si32(ab_vec) & 0xFF);
#else
        return 0;
#endif
    }
};

} // namespace simsimd
} // namespace av