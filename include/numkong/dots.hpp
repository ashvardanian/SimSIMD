/**
 *  @brief C++ bindings for multi-target dot-product kernels.
 *  @file include/numkong/dots.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 */
#ifndef NK_DOTS_HPP
#define NK_DOTS_HPP

#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "numkong/dot.h"
#include "numkong/dots.h"
#include "numkong/sets.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Estimates the memory requirements for packed B matrix.
 *  @param[in] row_count Number of rows in B (n)
 *  @param[in] depth Number of dimensions per row (k)
 *  @return Size in bytes for row-major B data plus stride metadata
 *
 *  @tparam in_type_ Input element type
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
NK_PUBLIC size_t dots_packed_size(size_t row_count, size_t depth) {
    constexpr bool simd = allow_simd_ == prefer_simd_k;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) return nk_dots_packed_size_f64(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) return nk_dots_packed_size_f32(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) return nk_dots_packed_size_f16(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) return nk_dots_packed_size_bf16(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) return nk_dots_packed_size_i8(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) return nk_dots_packed_size_u8(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd) return nk_dots_packed_size_e4m3(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd) return nk_dots_packed_size_e5m2(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && simd) return nk_dots_packed_size_e2m3(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && simd) return nk_dots_packed_size_e3m2(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd) return nk_dots_packed_size_u4(row_count, depth);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd) return nk_dots_packed_size_i4(row_count, depth);
    else {
        // We need enough space for the pointer to the original B matrix and its stride
        return sizeof(void *) + sizeof(size_t);
    }
}

/**
 *  @brief Packs matrix B into row-major form for efficient dots_packed access.
 *  @param[in] b Input matrix B in row-major form [row_count x depth]
 *  @param[in] row_count Number of rows in B (n)
 *  @param[in] depth Number of dimensions per row (k)
 *  @param[in] b_stride_in_bytes Stride between rows of B in bytes
 *  @param[out] b_packed Output buffer for packed row-major B with metadata
 *
 *  @tparam in_type_ Input element type
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
NK_PUBLIC void dots_pack(in_type_ const *b, size_t row_count, size_t depth, size_t b_stride_in_bytes, void *b_packed) {
    using raw_t = typename in_type_::raw_t;
    constexpr bool simd = allow_simd_ == prefer_simd_k;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_dots_pack_f64(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_dots_pack_f32(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_dots_pack_f16(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_dots_pack_bf16(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_dots_pack_i8(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd)
        nk_dots_pack_u8(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd)
        nk_dots_pack_e4m3(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd)
        nk_dots_pack_e5m2(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && simd)
        nk_dots_pack_e2m3(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && simd)
        nk_dots_pack_e3m2(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd)
        nk_dots_pack_u4(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd)
        nk_dots_pack_i4(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed);
    else {
        // Persist the pointer to the original B matrix and its stride
        char *b_packed_bytes = reinterpret_cast<char *>(b_packed);
        std::memcpy(b_packed_bytes, &b, sizeof(void *));
        std::memcpy(b_packed_bytes + sizeof(void *), &b_stride_in_bytes, sizeof(size_t));
    }
}

/**
 *  @brief Reference unpacked GEMM: C = A × Bᵀ (row-major A and B, B transposed).
 *
 *  This matches BLAS sgemm/dgemm with CblasNoTrans for A and CblasTrans for B.
 *  Useful as a reference implementation for validating BLAS/MKL/Accelerate.
 *
 *  @param a Matrix A [m x k] row-major
 *  @param b Matrix B [n x k] row-major (accessed as B^T)
 *  @param c Output matrix C [m x n] row-major
 *  @param row_count Rows of A and C (m)
 *  @param column_count Rows of B and columns of C (n)
 *  @param depth Columns of A and B (k)
 *  @param a_stride_in_bytes Stride between rows of A in bytes
 *  @param b_stride_in_bytes Stride between rows of B in bytes
 *  @param c_stride_in_bytes Stride between rows of C in bytes
 *  @tparam in_type_ Input element type (e.g., f32_t, bf16_t)
 *  @tparam result_type_ Accumulator/output type (e.g., f32_t, f118_t for high precision)
 */
template <typename in_type_, typename result_type_ = typename in_type_::dot_result_t>
void dots_unpacked(in_type_ const *a, in_type_ const *b, result_type_ *c, size_t row_count, size_t column_count,
                   size_t depth, size_t a_stride_in_bytes, size_t b_stride_in_bytes,
                   size_t c_stride_in_bytes) noexcept {
    char const *a_bytes = reinterpret_cast<char const *>(a);
    char const *b_bytes = reinterpret_cast<char const *>(b);
    char *c_bytes = reinterpret_cast<char *>(c);
    std::size_t const depth_values = divide_round_up(depth, dimensions_per_value<in_type_>());

    for (size_t i = 0; i < row_count; i++) {
        in_type_ const *a_row = reinterpret_cast<in_type_ const *>(a_bytes + i * a_stride_in_bytes);
        result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);
        for (size_t j = 0; j < column_count; j++) {
            in_type_ const *b_row = reinterpret_cast<in_type_ const *>(b_bytes + j * b_stride_in_bytes);
            result_type_ sum {};
            for (size_t l = 0; l < depth_values; l++) sum = fused_multiply_add(sum, a_row[l], b_row[l]);
            c_row[j] = sum;
        }
    }
}

/**
 *  @brief Packed dot products (batch matrix multiply): C = A × B (row-major)
 *  @param[in] a Matrix A [m x k]
 *  @param[in] b_packed Packed matrix B [k x n] with stride metadata appended
 *  @param[out] c Output matrix C [m x n]
 *  @param[in] row_count Rows of A and C (m)
 *  @param[in] column_count Columns of B and C (n)
 *  @param[in] depth Columns of A, Rows of B (k)
 *  @param[in] a_stride_in_bytes Stride between rows of A in bytes
 *  @param[in] c_stride_in_bytes Stride between rows of C in bytes
 *
 *  @tparam in_type_ Input element type
 *  @tparam result_type_ Accumulator/output type, defaults to `in_type_::dot_result_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename result_type_ = typename in_type_::dot_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void dots_packed(in_type_ const *a, void const *b_packed, result_type_ *c, size_t row_count, size_t column_count,
                 size_t depth, size_t a_stride_in_bytes, size_t c_stride_in_bytes) noexcept {
    using raw_t = typename in_type_::raw_t;
    constexpr bool dispatch = allow_simd_ == prefer_simd_k &&
                              std::is_same_v<result_type_, typename in_type_::dot_result_t>;
    if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
        nk_dots_packed_f64(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
        nk_dots_packed_f32(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
        nk_dots_packed_f16(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
        nk_dots_packed_bf16(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes,
                            c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
        nk_dots_packed_i8(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
        nk_dots_packed_u8(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
        nk_dots_packed_e4m3(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes,
                            c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
        nk_dots_packed_e5m2(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes,
                            c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
        nk_dots_packed_e2m3(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes,
                            c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
        nk_dots_packed_e3m2(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes,
                            c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
        nk_dots_packed_u4(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
        nk_dots_packed_i4(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    else {
        in_type_ const *b;
        size_t b_stride_in_bytes;
        char const *b_packed_bytes = reinterpret_cast<char const *>(b_packed);
        std::memcpy(&b, b_packed_bytes, sizeof(void *));
        std::memcpy(&b_stride_in_bytes, b_packed_bytes + sizeof(void *), sizeof(size_t));
        dots_unpacked<in_type_, result_type_>(a, b, c, row_count, column_count, depth, a_stride_in_bytes,
                                              b_stride_in_bytes, c_stride_in_bytes);
    }
}

/**
 *  @brief Symmetric dot products: C = A × Aᵀ where C[i,j] = ⟨A[i], A[j]⟩
 *  @param[in] a Matrix A [n x k] (n vectors of dimension k)
 *  @param[in] n_vectors Number of vectors (n)
 *  @param[in] depth Dimension of each vector (k)
 *  @param[in] a_stride_in_bytes Stride between vectors in A
 *  @param[out] c Output matrix C [n x n]
 *  @param[in] c_stride_in_bytes Stride between rows of C in bytes
 *
 *  @tparam in_type_ Input element type
 *  @tparam result_type_ Accumulator/output type, defaults to `in_type_::dot_result_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename result_type_ = typename in_type_::dot_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void dots_symmetric(in_type_ const *a, std::size_t n_vectors, std::size_t depth, std::size_t a_stride_in_bytes,
                    result_type_ *c, std::size_t c_stride_in_bytes, std::size_t row_start = 0,
                    std::size_t row_count = std::numeric_limits<std::size_t>::max()) noexcept {
    if (row_count == std::numeric_limits<std::size_t>::max()) row_count = n_vectors;
    using raw_t = typename in_type_::raw_t;
    constexpr bool dispatch = allow_simd_ == prefer_simd_k &&
                              std::is_same_v<result_type_, typename in_type_::dot_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
        nk_dots_symmetric_f64(&a->raw_, n_vectors, depth, a_stride_in_bytes, c, c_stride_in_bytes, row_start,
                              row_count);
    else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
        nk_dots_symmetric_f32(&a->raw_, n_vectors, depth, a_stride_in_bytes, c, c_stride_in_bytes, row_start,
                              row_count);
    else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
        nk_dots_symmetric_f16(&a->raw_, n_vectors, depth, a_stride_in_bytes, c, c_stride_in_bytes, row_start,
                              row_count);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
        nk_dots_symmetric_bf16(&a->raw_, n_vectors, depth, a_stride_in_bytes, c, c_stride_in_bytes, row_start,
                               row_count);
    else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
        nk_dots_symmetric_i8(&a->raw_, n_vectors, depth, a_stride_in_bytes, c, c_stride_in_bytes, row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
        nk_dots_symmetric_u8(&a->raw_, n_vectors, depth, a_stride_in_bytes, c, c_stride_in_bytes, row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
        nk_dots_symmetric_e4m3(&a->raw_, n_vectors, depth, a_stride_in_bytes, c, c_stride_in_bytes, row_start,
                               row_count);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
        nk_dots_symmetric_e5m2(&a->raw_, n_vectors, depth, a_stride_in_bytes, c, c_stride_in_bytes, row_start,
                               row_count);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
        nk_dots_symmetric_e2m3(&a->raw_, n_vectors, depth, a_stride_in_bytes, c, c_stride_in_bytes, row_start,
                               row_count);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
        nk_dots_symmetric_e3m2(&a->raw_, n_vectors, depth, a_stride_in_bytes, c, c_stride_in_bytes, row_start,
                               row_count);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
        nk_dots_symmetric_u4(&a->raw_, n_vectors, depth, a_stride_in_bytes, c, c_stride_in_bytes, row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
        nk_dots_symmetric_i4(&a->raw_, n_vectors, depth, a_stride_in_bytes, c, c_stride_in_bytes, row_start, row_count);
    else {
        std::size_t depth_values = divide_round_up(depth, dimensions_per_value<in_type_>());
        char const *a_bytes = reinterpret_cast<char const *>(a);
        char *c_bytes = reinterpret_cast<char *>(c);
        std::size_t row_end = row_start + row_count < n_vectors ? row_start + row_count : n_vectors;

        for (std::size_t i = row_start; i < row_end; i++) {
            in_type_ const *a_i = reinterpret_cast<in_type_ const *>(a_bytes + i * a_stride_in_bytes);
            result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);
            for (std::size_t j = 0; j < n_vectors; j++) {
                in_type_ const *a_j = reinterpret_cast<in_type_ const *>(a_bytes + j * a_stride_in_bytes);
                result_type_ sum {};
                for (std::size_t l = 0; l < depth_values; l++) sum = fused_multiply_add(sum, a_i[l], a_j[l]);
                c_row[j] = sum;
            }
        }
    }
}

/**
 *  @brief Estimates memory requirements for packed B matrix (Hamming distances).
 *  @param[in] row_count Number of rows in B (n)
 *  @param[in] depth Number of dimensions per row in bits (k)
 *  @return Size in bytes for row-major B data plus stride metadata
 *
 *  @tparam in_type_ Input element type (u1x8_t for binary vectors)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
NK_PUBLIC size_t hammings_packed_size(size_t row_count, size_t depth) {
    constexpr bool simd = allow_simd_ == prefer_simd_k;

    if constexpr (std::is_same_v<in_type_, u1x8_t> && simd) return nk_hammings_packed_size_u1(row_count, depth);
    else {
        // We need enough space for the pointer to the original B matrix and its stride
        return sizeof(void *) + sizeof(size_t);
    }
}

/**
 *  @brief Packs matrix B into row-major form for efficient hammings_packed access.
 *  @param[in] b Input matrix B in row-major form [row_count x depth]
 *  @param[in] row_count Number of rows in B (n)
 *  @param[in] depth Number of dimensions per row in bits (k)
 *  @param[in] b_stride_in_bytes Stride between rows of B in bytes
 *  @param[out] b_packed Output buffer for packed row-major B with metadata
 *
 *  @tparam in_type_ Input element type (u1x8_t for binary vectors)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
NK_PUBLIC void hammings_pack(in_type_ const *b, size_t row_count, size_t depth, size_t b_stride_in_bytes,
                             void *b_packed) {
    using raw_t = typename in_type_::raw_t;
    constexpr bool simd = allow_simd_ == prefer_simd_k;

    if constexpr (std::is_same_v<in_type_, u1x8_t> && simd)
        nk_hammings_pack_u1(reinterpret_cast<raw_t const *>(b), row_count, depth, b_stride_in_bytes, b_packed);
    else {
        // Persist the pointer to the original B matrix and its stride
        char *b_packed_bytes = reinterpret_cast<char *>(b_packed);
        std::memcpy(b_packed_bytes, &b, sizeof(void *));
        std::memcpy(b_packed_bytes + sizeof(void *), &b_stride_in_bytes, sizeof(size_t));
    }
}

/**
 *  @brief Symmetric Hamming distance matrix: C[i,j] = hamming(A[i], A[j])
 *  @param[in] a Input matrix (n_vectors x depth)
 *  @param[in] n_vectors Number of vectors
 *  @param[in] depth Number of dimensions per vector
 *  @param[in] a_stride_in_bytes Row stride in bytes
 *  @param[out] c Output matrix (n_vectors x n_vectors)
 *  @param[in] c_stride_in_bytes Output row stride in bytes
 *  @param[in] row_start Starting row index (default 0)
 *  @param[in] row_count Number of rows to compute (default all)
 *
 *  Computes Hamming distances between all pairs of binary vectors.
 *  For u1x8_t inputs, distances are exact bit counts (u32_t outputs).
 *
 *  @tparam in_type_ Input element type (u1x8_t)
 *  @tparam result_type_ Output type (u32_t for Hamming distances)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename result_type_ = u32_t, allow_simd_t allow_simd_ = prefer_simd_k>
void hammings_symmetric(in_type_ const *a, std::size_t n_vectors, std::size_t depth, std::size_t a_stride_in_bytes,
                        result_type_ *c, std::size_t c_stride_in_bytes, std::size_t row_start = 0,
                        std::size_t row_count = std::numeric_limits<std::size_t>::max()) noexcept {
    if (row_count == std::numeric_limits<std::size_t>::max()) row_count = n_vectors;
    using raw_t = typename in_type_::raw_t;
    constexpr bool dispatch = allow_simd_ == prefer_simd_k && std::is_same_v<in_type_, u1x8_t> &&
                              std::is_same_v<result_type_, u32_t>;

    if constexpr (dispatch)
        nk_hammings_symmetric_u1(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes, row_start,
                                 row_count);
    else {
        std::size_t depth_bytes = divide_round_up(depth, 8);
        char const *a_bytes = reinterpret_cast<char const *>(a);
        char *c_bytes = reinterpret_cast<char *>(c);
        std::size_t row_end = row_start + row_count < n_vectors ? row_start + row_count : n_vectors;

        for (std::size_t i = row_start; i < row_end; i++) {
            raw_t const *a_i = reinterpret_cast<raw_t const *>(a_bytes + i * a_stride_in_bytes);
            result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);

            for (std::size_t j = 0; j < n_vectors; j++) {
                raw_t const *a_j = reinterpret_cast<raw_t const *>(a_bytes + j * a_stride_in_bytes);
                typename result_type_::raw_t distance = 0;
                for (std::size_t b = 0; b < depth_bytes; b++) {
                    auto xor_val = a_i[b] ^ a_j[b];
                    distance += std::popcount(static_cast<unsigned>(xor_val));
                }
                c_row[j] = result_type_::from_raw(distance);
            }
        }
    }
}

/**
 *  @brief Computes Hamming distances between rows of A and columns of packed B.
 *  @param[in] a Pointer to the first matrix (m x k).
 *  @param[in] b_packed Pointer to the packed second matrix (k x n).
 *  @param[out] c Pointer to the output matrix (m x n).
 *  @param[in] row_count Number of rows in A (m).
 *  @param[in] column_count Number of columns in B (n).
 *  @param[in] depth Depth dimension in bits (k).
 *  @param[in] a_stride_in_bytes Stride between consecutive rows of A in bytes.
 *  @param[in] c_stride_in_bytes Stride between consecutive rows of C in bytes.
 *
 *  Computes Hamming distances between binary vectors using optimized packed format.
 *  For u1x8_t inputs, distances are exact bit counts (u32_t outputs).
 *
 *  @tparam in_type_ Input element type (u1x8_t)
 *  @tparam result_type_ Output type (u32_t for Hamming distances)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename result_type_ = u32_t, allow_simd_t allow_simd_ = prefer_simd_k>
void hammings_packed(in_type_ const *a, void const *b_packed, result_type_ *c, std::size_t row_count,
                     std::size_t column_count, std::size_t depth, std::size_t a_stride_in_bytes = 0,
                     std::size_t c_stride_in_bytes = 0) noexcept {
    // Compute default strides
    if (!a_stride_in_bytes) a_stride_in_bytes = divide_round_up(depth, 8) * sizeof(in_type_);
    if (!c_stride_in_bytes) c_stride_in_bytes = column_count * sizeof(result_type_);

    // SIMD dispatch for u1x8_t -> u32_t
    if constexpr (allow_simd_ && std::is_same_v<in_type_, u1x8_t> && std::is_same_v<result_type_, u32_t>) {
        nk_hammings_packed_u1(reinterpret_cast<nk_u1x8_t const *>(a), b_packed, reinterpret_cast<nk_u32_t *>(c),
                              row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    }
    else {

        // Scalar fallback: extract pointer and stride from b_packed, then compute directly
        in_type_ const *b;
        size_t b_stride_in_bytes;
        char const *b_packed_bytes = reinterpret_cast<char const *>(b_packed);
        std::memcpy(&b, b_packed_bytes, sizeof(void *));
        std::memcpy(&b_stride_in_bytes, b_packed_bytes + sizeof(void *), sizeof(size_t));

        // Compute Hamming distances using unpacked matrices
        char const *a_bytes = reinterpret_cast<char const *>(a);
        char const *b_bytes = reinterpret_cast<char const *>(b);
        char *c_bytes = reinterpret_cast<char *>(c);
        std::size_t depth_bytes = divide_round_up(depth, 8);

        for (std::size_t i = 0; i < row_count; i++) {
            typename in_type_::raw_t const *a_row = reinterpret_cast<typename in_type_::raw_t const *>(
                a_bytes + i * a_stride_in_bytes);
            result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);

            for (std::size_t j = 0; j < column_count; j++) {
                typename in_type_::raw_t const *b_row = reinterpret_cast<typename in_type_::raw_t const *>(
                    b_bytes + j * b_stride_in_bytes);

                // Compute Hamming distance: XOR then popcount
                typename result_type_::raw_t distance = 0;
                for (std::size_t byte_idx = 0; byte_idx < depth_bytes; byte_idx++) {
                    auto xor_val = a_row[byte_idx] ^ b_row[byte_idx];
                    distance += std::popcount(static_cast<unsigned>(xor_val));
                }
                c_row[j] = result_type_::from_raw(distance);
            }
        }
    }
}

} // namespace ashvardanian::numkong

#endif // NK_DOTS_HPP
