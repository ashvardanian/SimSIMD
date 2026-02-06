/**
 *  @brief C++ bindings for set-intersection kernels.
 *  @file include/numkong/set.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 */
#ifndef NK_SET_HPP
#define NK_SET_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/set.h"
#include "numkong/sets.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Hamming distance: Σ(aᵢ ⊕ bᵢ)
 *  @param[in] a,b Input vectors
 *  @param[in] d Number of dimensions (elements for u8_t, bits/8 for u1x8_t)
 *  @param[out] r Pointer to output count
 *
 *  @tparam in_type_ Input vector element type (u1x8_t or u8_t)
 *  @tparam result_type_ Accumulator type, defaults to `in_type_::hamming_result_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename result_type_ = typename in_type_::hamming_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void hamming(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<result_type_, typename in_type_::hamming_result_t>;

    if constexpr (std::is_same_v<in_type_, u1x8_t> && simd) nk_hamming_u1(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_hamming_u8(&a->raw_, &b->raw_, d, &r->raw_);
    // Scalar fallback
    else {
        typename result_type_::raw_t count = 0;
        for (std::size_t i = 0; i < d; i++) count += (a[i] != b[i]) ? 1 : 0;
        *r = result_type_::from_raw(count);
    }
}

/**
 *  @brief Jaccard distance: 1 − |A ∩ B| / |A ∪ B|
 *  @param[in] a,b Input vectors
 *  @param[in] d Number of dimensions
 *  @param[out] r Pointer to output distance
 *
 *  For u1x8_t (bit vectors): uses |A intersect B| / |A union B| (set intersection/union)
 *  For u16_t/u32_t (element vectors): uses count of matching elements / total
 *
 *  @tparam in_type_ Input vector element type (u1x8_t, u16_t, or u32_t)
 *  @tparam result_type_ Accumulator type, defaults to `in_type_::jaccard_result_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename result_type_ = typename in_type_::jaccard_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void jaccard(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<result_type_, typename in_type_::jaccard_result_t>;

    if constexpr (std::is_same_v<in_type_, u1x8_t> && simd) nk_jaccard_u1(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd) nk_jaccard_u16(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd) nk_jaccard_u32(&a->raw_, &b->raw_, d, &r->raw_);
    // Scalar fallback for u1x8_t (bit vectors)
    else if constexpr (std::is_same_v<in_type_, u1x8_t>) {
        std::uint32_t intersection_count = 0, union_count = 0;
        for (std::size_t i = 0; i < d; i++)
            intersection_count += a[i].intersection(b[i]), union_count += a[i].union_size(b[i]);
        if (union_count == 0) *r = result_type_(0.0f);
        else *r = result_type_(1.0f - static_cast<float>(intersection_count) / static_cast<float>(union_count));
    }
    // Scalar fallback for element vectors
    else {
        std::size_t matches = 0;
        for (std::size_t i = 0; i < d; i++) matches += (a[i] == b[i]) ? 1 : 0;
        if (d == 0) *r = result_type_(1.0f);
        else *r = result_type_(1.0f - static_cast<float>(matches) / static_cast<float>(d));
    }
}

} // namespace ashvardanian::numkong

#endif // NK_SET_HPP
