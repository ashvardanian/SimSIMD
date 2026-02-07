/**
 *  @brief Reduction kernels: reduce_add, reduce_min, reduce_max.
 *  @file include/numkong/reduce.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 */
#ifndef NK_REDUCE_HPP
#define NK_REDUCE_HPP

#include <cstddef>     // `std::byte`, `std::size_t`
#include <cstdint>     // `std::uint32_t`
#include <type_traits> // `std::is_same_v`

#include "numkong/reduce.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Sum all elements: Σ dataᵢ
 *  @param[in] data Input array
 *  @param[in] count Number of elements
 *  @param[in] stride_bytes Stride between elements in bytes (use sizeof(in_type_) for contiguous)
 *  @param[out] result Output sum
 *
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Accumulator type, defaults to `in_type_::reduce_add_result_t` (often widened)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename result_type_ = typename in_type_::reduce_add_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void reduce_add(in_type_ const *data, std::size_t count, std::size_t stride_bytes, result_type_ *result) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<result_type_, typename in_type_::reduce_add_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_reduce_add_f64(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_reduce_add_f32(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_reduce_add_f16(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_reduce_add_bf16(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd)
        nk_reduce_add_e4m3(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd)
        nk_reduce_add_e5m2(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && simd)
        nk_reduce_add_e2m3(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && simd)
        nk_reduce_add_e3m2(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd)
        nk_reduce_add_i4(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd)
        nk_reduce_add_u4(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, u1x8_t> && simd)
        nk_reduce_add_u1(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_reduce_add_i8(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd)
        nk_reduce_add_u8(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd)
        nk_reduce_add_i16(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd)
        nk_reduce_add_u16(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd)
        nk_reduce_add_i32(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd)
        nk_reduce_add_u32(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd)
        nk_reduce_add_i64(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd)
        nk_reduce_add_u64(&data->raw_, count, stride_bytes, &result->raw_);
    // Scalar fallback
    else {
        result_type_ sum {};
        auto const *ptr = reinterpret_cast<std::byte const *>(data);
        for (std::size_t i = 0; i < count; i++, ptr += stride_bytes)
            sum = sum.saturating_add(result_type_(*reinterpret_cast<in_type_ const *>(ptr)));
        *result = sum;
    }
}

/**
 *  @brief Find minimum element and its index: argmin dataᵢ
 *  @param[in] data Input array
 *  @param[in] count Number of elements
 *  @param[in] stride_bytes Stride between elements in bytes (use sizeof(in_type_) for contiguous)
 *  @param[out] min_value Output minimum value
 *  @param[out] min_index Output index of minimum value
 *
 *  @tparam in_type_ Input vector element type
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void reduce_min(in_type_ const *data, std::size_t count, std::size_t stride_bytes, in_type_ *min_value,
                std::size_t *min_index) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_reduce_min_f64(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_reduce_min_f32(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_reduce_min_f16(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_reduce_min_bf16(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd)
        nk_reduce_min_e4m3(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd)
        nk_reduce_min_e5m2(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && simd)
        nk_reduce_min_e2m3(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && simd)
        nk_reduce_min_e3m2(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd)
        nk_reduce_min_i4(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd)
        nk_reduce_min_u4(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, u1x8_t> && simd)
        nk_reduce_min_u1(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_reduce_min_i8(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd)
        nk_reduce_min_u8(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd)
        nk_reduce_min_i16(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd)
        nk_reduce_min_u16(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd)
        nk_reduce_min_i32(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd)
        nk_reduce_min_u32(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd)
        nk_reduce_min_i64(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd)
        nk_reduce_min_u64(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    // Scalar fallback
    else {
        in_type_ best_value = in_type_::finite_max();
        std::size_t best_index = 0;
        auto const *ptr = reinterpret_cast<std::byte const *>(data);
        for (std::size_t i = 0; i < count; i++, ptr += stride_bytes) {
            in_type_ v = *reinterpret_cast<in_type_ const *>(ptr);
            if (v < best_value) best_value = v, best_index = i;
        }
        *min_value = best_value;
        *min_index = best_index;
    }
}

/**
 *  @brief Find maximum element and its index: argmax dataᵢ
 *  @param[in] data Input array
 *  @param[in] count Number of elements
 *  @param[in] stride_bytes Stride between elements in bytes (use sizeof(in_type_) for contiguous)
 *  @param[out] max_value Output maximum value
 *  @param[out] max_index Output index of maximum value
 *
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Result type for the maximum value, defaults to `in_type_`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename result_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void reduce_max(in_type_ const *data, std::size_t count, std::size_t stride_bytes, result_type_ *max_value,
                std::size_t *max_index) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_reduce_max_f64(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_reduce_max_f32(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_reduce_max_f16(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_reduce_max_bf16(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd)
        nk_reduce_max_e4m3(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd)
        nk_reduce_max_e5m2(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && simd)
        nk_reduce_max_e2m3(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && simd)
        nk_reduce_max_e3m2(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd)
        nk_reduce_max_i4(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd)
        nk_reduce_max_u4(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, u1x8_t> && simd)
        nk_reduce_max_u1(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_reduce_max_i8(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd)
        nk_reduce_max_u8(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd)
        nk_reduce_max_i16(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd)
        nk_reduce_max_u16(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd)
        nk_reduce_max_i32(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd)
        nk_reduce_max_u32(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd)
        nk_reduce_max_i64(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd)
        nk_reduce_max_u64(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    // Scalar fallback
    else {
        in_type_ best_value = in_type_::finite_min();
        std::size_t best_index = 0;
        auto const *ptr = reinterpret_cast<std::byte const *>(data);
        for (std::size_t i = 0; i < count; i++, ptr += stride_bytes) {
            in_type_ v = *reinterpret_cast<in_type_ const *>(ptr);
            if (v > best_value) best_value = v, best_index = i;
        }
        *max_value = best_value;
        *max_index = best_index;
    }
}

} // namespace ashvardanian::numkong

#endif // NK_REDUCE_HPP
