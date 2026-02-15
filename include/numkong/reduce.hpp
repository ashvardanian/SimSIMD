/**
 *  @brief Reduction kernels: reduce_moments (sum + sum-of-squares), reduce_minmax (min + max with indices).
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
 *  @brief Compute sum and sum-of-squares in a single pass: sum = Sigma data_i, sumsq = Sigma data_i^2
 *  @param[in] data Input array
 *  @param[in] count Number of elements
 *  @param[in] stride_bytes Stride between elements in bytes (use sizeof(in_type_) for contiguous)
 *  @param[out] sum Output sum
 *  @param[out] sumsq Output sum of squares
 *
 *  @tparam in_type_ Input vector element type
 *  @tparam sum_type_ Sum accumulator type, defaults to `in_type_::reduce_moments_sum_t` (often widened)
 *  @tparam sumsq_type_ Sum-of-squares accumulator type, defaults to `sum_type_`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename sum_type_ = typename in_type_::reduce_moments_sum_t,
          typename sumsq_type_ = typename in_type_::reduce_moments_sumsq_t, allow_simd_t allow_simd_ = prefer_simd_k>
void reduce_moments(in_type_ const *data, std::size_t count, std::size_t stride_bytes, sum_type_ *sum,
                    sumsq_type_ *sumsq) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<sum_type_, typename in_type_::reduce_moments_sum_t> &&
                          std::is_same_v<sumsq_type_, typename in_type_::reduce_moments_sumsq_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_reduce_moments_f64(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_reduce_moments_f32(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_reduce_moments_f16(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_reduce_moments_bf16(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd)
        nk_reduce_moments_e4m3(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd)
        nk_reduce_moments_e5m2(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && simd)
        nk_reduce_moments_e2m3(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && simd)
        nk_reduce_moments_e3m2(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd)
        nk_reduce_moments_i4(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd)
        nk_reduce_moments_u4(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, u1x8_t> && simd)
        nk_reduce_moments_u1(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_reduce_moments_i8(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd)
        nk_reduce_moments_u8(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd)
        nk_reduce_moments_i16(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd)
        nk_reduce_moments_u16(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd)
        nk_reduce_moments_i32(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd)
        nk_reduce_moments_u32(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd)
        nk_reduce_moments_i64(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd)
        nk_reduce_moments_u64(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    // Scalar fallback
    else {
        sum_type_ running_sum {};
        sumsq_type_ running_sumsq {};
        auto const *ptr = reinterpret_cast<std::byte const *>(data);
        for (std::size_t i = 0; i < count; i++, ptr += stride_bytes) {
            sum_type_ val(static_cast<sum_type_>(*reinterpret_cast<in_type_ const *>(ptr)));
            running_sum = running_sum.saturating_add(val);
            running_sumsq = running_sumsq.saturating_add(static_cast<sumsq_type_>(val.saturating_mul(val)));
        }
        *sum = running_sum;
        *sumsq = running_sumsq;
    }
}

/**
 *  @brief Find minimum and maximum elements with their indices in a single pass.
 *  @param[in] data Input array
 *  @param[in] count Number of elements
 *  @param[in] stride_bytes Stride between elements in bytes (use sizeof(in_type_) for contiguous)
 *  @param[out] min_value Output minimum value
 *  @param[out] min_index Output index of minimum value
 *  @param[out] max_value Output maximum value
 *  @param[out] max_index Output index of maximum value
 *
 *  @tparam in_type_ Input vector element type
 *  @tparam minmax_type_ Result type for min/max values, defaults to `in_type_::reduce_minmax_value_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename minmax_type_ = typename in_type_::reduce_minmax_value_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void reduce_minmax(in_type_ const *data, std::size_t count, std::size_t stride_bytes, minmax_type_ *min_value,
                   std::size_t *min_index, minmax_type_ *max_value, std::size_t *max_index) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<minmax_type_, typename in_type_::reduce_minmax_value_t>;

    // For types where minmax_type_ matches the C function output type directly,
    // dispatch to the C kernel and pass raw pointers through.
    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_reduce_minmax_f64(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_,
                             max_index);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_reduce_minmax_f32(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_,
                             max_index);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_reduce_minmax_i8(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd)
        nk_reduce_minmax_u8(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd)
        nk_reduce_minmax_i16(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_,
                             max_index);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd)
        nk_reduce_minmax_u16(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_,
                             max_index);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd)
        nk_reduce_minmax_i32(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_,
                             max_index);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd)
        nk_reduce_minmax_u32(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_,
                             max_index);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd)
        nk_reduce_minmax_i64(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_,
                             max_index);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd)
        nk_reduce_minmax_u64(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_,
                             max_index);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && simd)
        nk_reduce_minmax_e2m3(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_,
                              max_index);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && simd)
        nk_reduce_minmax_e3m2(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_,
                              max_index);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_reduce_minmax_f16(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_,
                             max_index);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_reduce_minmax_bf16(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_,
                              max_index);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd)
        nk_reduce_minmax_e4m3(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_,
                              max_index);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd)
        nk_reduce_minmax_e5m2(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_,
                              max_index);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd)
        nk_reduce_minmax_i4(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd)
        nk_reduce_minmax_u4(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, u1x8_t> && simd)
        nk_reduce_minmax_u1(&data->raw_, count, stride_bytes, &min_value->raw_, min_index, &max_value->raw_, max_index);
    // Scalar fallback
    else {
        minmax_type_ best_min = minmax_type_(in_type_::finite_max());
        minmax_type_ best_max = minmax_type_(in_type_::finite_min());
        std::size_t best_min_index = 0;
        std::size_t best_max_index = 0;
        auto const *ptr = reinterpret_cast<std::byte const *>(data);
        for (std::size_t i = 0; i < count; i++, ptr += stride_bytes) {
            minmax_type_ v = minmax_type_(*reinterpret_cast<in_type_ const *>(ptr));
            if (v < best_min) best_min = v, best_min_index = i;
            if (v > best_max) best_max = v, best_max_index = i;
        }
        *min_value = best_min, *min_index = best_min_index;
        *max_value = best_max, *max_index = best_max_index;
    }
}

} // namespace ashvardanian::numkong

#endif // NK_REDUCE_HPP
