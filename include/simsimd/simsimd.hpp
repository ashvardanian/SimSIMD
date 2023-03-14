/**
 * x86: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 * Arm: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 */

#pragma once
#include <cmath>   // `std::sqrt`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::int8_t`

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

namespace av::simsimd {

using i8_t = std::int8_t;
using i16_t = std::int16_t;
using i32_t = std::int32_t;
using f32_t = float;
using f64_t = double;
enum class f16_t : std::int16_t {};

union f32i32_t {
    std::int32_t i;
    float f;
};

struct dot_product_t {
    template <typename at> at operator()(at const* a, at const* b, std::size_t const dim) const noexcept {
        at dist = 0;
#pragma GCC ivdep
#pragma clang loop vectorize(enable)
        for (std::size_t i = 0; i != dim; ++i)
            dist += a[i] * b[i];
        return 1 - dist;
    }
};

struct cosine_similarity_t {
    template <typename at> at operator()(at const* a, at const* b, std::size_t const dim) const noexcept {
        at dist = 0, a_sq = 0, b_sq = 0;
#pragma GCC ivdep
#pragma clang loop vectorize(enable)
        for (std::size_t i = 0; i != dim; ++i)
            dist += a[i] * b[i], a_sq += a[i] * a[i], b_sq += b[i] * b[i];
        return 1 - dist / (std::sqrt(a_sq) * std::sqrt(b_sq));
    }
};

struct euclidean_distance_t {
    f32_t operator()(i8_t const* a, i8_t const* b, std::size_t const dim) const noexcept {
        i32_t dist = 0;
#pragma GCC ivdep
#pragma clang loop vectorize(enable)
        for (std::size_t i = 0; i != dim; ++i)
            dist += i32_t(i16_t(a[i]) - i16_t(b[i])) * i32_t(i16_t(a[i]) - i16_t(b[i]));
        return std::sqrt(dist);
    }
    template <typename at> at operator()(at const* a, at const* b, std::size_t const dim) const noexcept {
        at dist = 0;
#pragma GCC ivdep
#pragma clang loop vectorize(enable)
        for (std::size_t i = 0; i != dim; ++i)
            dist += (a[i] - b[i]) * (a[i] - b[i]);
        return std::sqrt(dist);
    }
};

/**
 * @brief
 *      SIMD-accelerated dot-product distance, that assumes vector sizes to be multiples
 *      of 128 bits, and address of the first argument to be aligned to the same size registers.
 */
struct dot_product_f32x4k_t {

    f32_t operator()(f32_t const* a, f32_t const* b, std::size_t const dim) const noexcept {
#if defined(__AVX2__)
        __m128 abs = _mm_set1_ps(0);
        for (std::size_t i = 0; i != dim; i += 4)
            abs = _mm_fmadd_ps(_mm_load_ps(a + i), _mm_loadu_ps(b + i), abs);
        abs = _mm_hadd_ps(abs, abs);
        abs = _mm_hadd_ps(abs, abs);
        f32i32_t abs_u = {_mm_cvtsi128_si32(_mm_castps_si128(abs))};
        return 1 - abs_u.f;

#elif defined(__ARM_FEATURE_SVE)
        int64_t i = 0;
        svfloat32_t abs_vec;
        svbool_t pg = svwhilelt_b32(i, n);                                           
        do{
                        svfloat32_t ai_vec = svld1_f32(pg, a+i);                     
                        svfloat32_t bi_vec = svld1_f32(pg, b+i);                     
                        abs_vec = svmla_f32_x(pg, abs, ai_vec, bi_vec);         
                        i += svcntw();                                              
                        pg = svwhilelt_b32(i, n);                                   
                }
        while (svptest_any(svptrue_b64(), pg));                                   

#elif defined(__ARM_NEON__)
        float32x4_t abs = vdupq_n_f32(0);
        for (std::size_t i = 0; i != dim; i += 4)
            abs = vmlaq_f32(abs, vld1q_f32(a + i), vld1q_f32(b + i));
        return 1 - vaddvq_f32(abs);
#endif
    }
};

struct cosine_similarity_f32x4k_t {

    f32_t operator()(f32_t const* a, f32_t const* b, std::size_t const dim) const noexcept {
#if defined(__AVX2__)
        __m128 abs = _mm_set1_ps(0);
        __m128 as_sq = _mm_set1_ps(0);
        __m128 bs_sq = _mm_set1_ps(0);
        for (std::size_t i = 0; i != dim; i += 4) {
            auto ai = _mm_load_ps(a + i);
            auto bi = _mm_loadu_ps(b + i);
            abs = _mm_fmadd_ps(ai, bi, abs);
            as_sq = _mm_fmadd_ps(ai, ai, as_sq);
            bs_sq = _mm_fmadd_ps(bi, bi, bs_sq);
        }
        abs = _mm_hadd_ps(abs, abs);
        abs = _mm_hadd_ps(abs, abs);
        as_sq = _mm_hadd_ps(as_sq, as_sq);
        as_sq = _mm_hadd_ps(as_sq, as_sq);
        bs_sq = _mm_hadd_ps(bs_sq, bs_sq);
        bs_sq = _mm_hadd_ps(bs_sq, bs_sq);
        f32i32_t abs_u = {_mm_cvtsi128_si32(_mm_castps_si128(abs))};
        f32i32_t as_sq_u = {_mm_cvtsi128_si32(_mm_castps_si128(as_sq))};
        f32i32_t bs_sq_u = {_mm_cvtsi128_si32(_mm_castps_si128(bs_sq))};
        return 1 - abs_u.f / (std::sqrt(as_sq_u.f) * std::sqrt(bs_sq_u.f));
#elif defined(__ARM_NEON__)
        float32x4_t abs = vdupq_n_f32(0);
        for (std::size_t i = 0; i != dim; i += 4)
            abs = vmlaq_f32(abs, vld1q_f32(a + i), vld1q_f32(b + i));
        return 1 - vaddvq_f32(abs);
#endif
    }
};

/** @brief Takes scalars quantized into a [-100; 100] interval. */
struct dot_product_i8x16k_t {

    i32_t operator()(i8_t const* a, i8_t const* b, std::size_t const dim) const noexcept {
#if defined(__AVX2__)
        __m256i abs = _mm256_set1_epi16(0);
        for (std::size_t i = 0; i != dim; i += 4)
            abs = _mm256_add_epi16(                                                //
                abs,                                                               //
                _mm256_mullo_epi16(                                                //
                    _mm256_cvtepi8_epi16(_mm_load_si128((__m128i const*)(a + i))), //
                    _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i const*)(b + i)))));
        abs = _mm256_hadd_epi16(abs, abs);
        abs = _mm256_hadd_epi16(abs, abs);
        abs = _mm256_hadd_epi16(abs, abs);
        return 1 - (_mm256_cvtsi256_si32(abs) & 0xFF);
#endif
    }
};



} // namespace unum::unsw