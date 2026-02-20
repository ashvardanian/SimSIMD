/**
 *  @brief Template test functions for batch operations (dots, hammings).
 *  @file test/test_cross.hpp
 *  @author Ash Vardanian
 *  @date January 14, 2025
 *
 *  This header contains the template test implementations that are shared
 *  across all ISA-specific test files.
 */
#pragma once
#ifndef NK_TEST_CROSS_HPP
#define NK_TEST_CROSS_HPP

#include "numkong/dots.hpp" // `nk::dots_packed_size`, `nk::dots_pack`, etc.
#include "test.hpp"

/**
 *  @brief Generic GEMM test against f118_t reference.
 *  Works for all types: f32, f64, f16, bf16, i8.
 */
template <typename scalar_type_>
error_stats_t test_dots(typename scalar_type_::dots_packed_size_kernel_t packed_size_fn,
                        typename scalar_type_::dots_pack_kernel_t pack_fn,
                        typename scalar_type_::dots_packed_kernel_t dots_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::dot_result_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);

    std::size_t m = global_config.matrix_height, n = global_config.matrix_width, k = global_config.matrix_depth;
    std::size_t k_values = nk::divide_round_up(k, nk::dimensions_per_value<scalar_t>());
    std::size_t a_stride = k_values * sizeof(scalar_t);
    std::size_t b_stride = k_values * sizeof(scalar_t);
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(m * k), b = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(m * n);
    auto c_ref = make_vector<f118_t>(m * n);

    nk_size_t packed_size = packed_size_fn(n, k);
    auto b_packed = make_vector<char>(packed_size);
    nk_size_t ref_packed_size = nk::dots_packed_size<scalar_t, nk::no_simd_k>(n, k);
    auto b_packed_ref = make_vector<char>(ref_packed_size);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        // Run kernel being tested
        pack_fn(b.raw_values_data(), n, k, b_stride, b_packed.raw_values_data());
        dots_fn(a.raw_values_data(), b_packed.raw_values_data(), c.raw_values_data(), m, n, k, a_stride, c_stride);

        // Compute f118_t reference using nk:: template
        nk::dots_pack<scalar_t, nk::no_simd_k>(b.values_data(), n, k, b_stride, b_packed_ref.raw_values_data());
        nk::dots_packed<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), b_packed_ref.raw_values_data(),
                                                         c_ref.raw_values_data(), m, n, k, a_stride,
                                                         n * sizeof(f118_t));

        for (std::size_t i = 0; i < m * n; i++) stats.accumulate(c[i], c_ref[i]);
    }
    return stats;
}

/**
 *  @brief Generic symmetric GEMM (A x A^T) test against f118_t reference.
 *  Works for all types: f32, f64, f16, bf16, i8, u8, i4, u4, e4m3, e5m2.
 */
template <typename scalar_type_>
error_stats_t test_dots_symmetric(typename scalar_type_::dots_symmetric_kernel_t kernel_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::dot_result_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);

    std::size_t n = global_config.matrix_height, k = global_config.matrix_depth;
    std::size_t k_values = nk::divide_round_up(k, nk::dimensions_per_value<scalar_t>());
    std::size_t a_stride = k_values * sizeof(scalar_t);
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(n * n);
    auto c_ref = make_vector<f118_t>(n * n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);

        // Run kernel being tested
        kernel_fn(a.raw_values_data(), n, k, a_stride, c.raw_values_data(), c_stride, 0, n);

        // Compute f118_t reference using nk:: template
        nk::dots_symmetric<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), n, k, a_stride, c_ref.raw_values_data(),
                                                            n * sizeof(f118_t));

        // Only check upper triangle and diagonal
        for (std::size_t i = 0; i < n; i++)
            for (std::size_t j = i; j < n; j++) stats.accumulate(c[i * n + j], c_ref[i * n + j]);
    }
    return stats;
}

/**
 *  @brief Test batched Hamming distance computation with packed B matrix.
 */
template <typename scalar_type_>
error_stats_t test_hammings(typename scalar_type_::hammings_packed_size_kernel_t packed_size_fn,
                            typename scalar_type_::hammings_pack_kernel_t pack_fn,
                            typename scalar_type_::hammings_packed_kernel_t hammings_fn) {
    using scalar_t = scalar_type_;
    using result_t = u32_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);

    std::size_t m = global_config.matrix_height, n = global_config.matrix_width, k = global_config.dense_dimensions;
    std::size_t k_bytes = nk::divide_round_up(k, 8);
    std::size_t a_stride = k_bytes * sizeof(scalar_t);
    std::size_t b_stride = k_bytes * sizeof(scalar_t);
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(m * k_bytes), b = make_vector<scalar_t>(n * k_bytes);
    auto c = make_vector<result_t>(m * n);
    auto c_ref = make_vector<result_t>(m * n);

    nk_size_t packed_size = packed_size_fn(n, k);
    auto b_packed = make_vector<char>(packed_size);

    // Allocate buffer for reference computation
    nk_size_t packed_size_ref = nk::hammings_packed_size<scalar_t, nk::no_simd_k>(n, k);
    auto b_packed_ref = make_vector<char>(packed_size_ref);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        // Run kernel being tested
        pack_fn(b.raw_values_data(), n, k, b_stride, b_packed.raw_values_data());
        hammings_fn(a.raw_values_data(), b_packed.raw_values_data(), c.raw_values_data(), m, n, k, a_stride, c_stride);

        // Compute reference using C++ template with no_simd_k
        nk::hammings_pack<scalar_t, nk::no_simd_k>(b.values_data(), n, k, b_stride, b_packed_ref.raw_values_data());
        nk::hammings_packed<scalar_t, result_t, nk::no_simd_k>(a.values_data(), b_packed_ref.raw_values_data(),
                                                               c_ref.values_data(), m, n, k, a_stride, c_stride);

        // Hamming distances are exact integers
        for (std::size_t i = 0; i < m * n; i++) stats.accumulate(c[i], c_ref[i]);
    }
    return stats;
}

/**
 *  @brief Test symmetric Hamming distance matrix computation.
 */
template <typename scalar_type_>
error_stats_t test_hammings_symmetric(typename scalar_type_::hammings_symmetric_kernel_t kernel_fn) {
    using scalar_t = scalar_type_;
    using result_t = u32_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);

    std::size_t n = global_config.matrix_height, k = global_config.dense_dimensions;
    std::size_t k_bytes = nk::divide_round_up(k, 8);
    std::size_t a_stride = k_bytes * sizeof(scalar_t);
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(n * k_bytes);
    auto c = make_vector<result_t>(n * n);
    auto c_ref = make_vector<result_t>(n * n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);

        // Run kernel being tested
        kernel_fn(a.raw_values_data(), n, k, a_stride, c.raw_values_data(), c_stride, 0, n);

        // Compute reference using nk:: template
        nk::hammings_symmetric<scalar_t, result_t, nk::no_simd_k>(a.values_data(), n, k, a_stride, c_ref.values_data(),
                                                                  n * sizeof(result_t));

        // Hamming distances are exact integers — check upper triangle only
        for (std::size_t i = 0; i < n; i++)
            for (std::size_t j = i; j < n; j++) stats.accumulate(c[i * n + j], c_ref[i * n + j]);
    }
    return stats;
}

// Forward declarations for cross functions
void test_cross_serial();
void test_cross_x86();
void test_cross_amx();
void test_cross_arm();
void test_cross_sme();
void test_cross_blas();

#endif // NK_TEST_CROSS_HPP
