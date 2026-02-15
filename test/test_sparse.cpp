/**
 *  @brief Sparse operations tests.
 *  @file test/test_sparse.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "test.hpp"
#include "numkong/sparse.hpp"

/**
 *  @brief Generate a sorted array of unique random indices (in-place).
 */
template <typename scalar_type_, typename generator_type_>
void fill_sorted_unique(generator_type_ &generator, nk::vector<scalar_type_> &vector, scalar_type_ max_val) {
    using raw_t = typename scalar_type_::raw_t;
    std::uniform_int_distribution<raw_t> distribution(0, static_cast<raw_t>(max_val));
    std::size_t const count = vector.size_values();

    // Fill and sort once
    for (std::size_t i = 0; i < count; ++i) vector[i] = scalar_type_(distribution(generator));
    std::sort(vector.values_data(), vector.values_data() + count);

    // Compact duplicates; refill gaps until full
    auto unique_end = std::unique(vector.values_data(), vector.values_data() + count);
    std::size_t unique_count = static_cast<std::size_t>(unique_end - vector.values_data());

    while (unique_count < count) {
        for (std::size_t i = unique_count; i < count; ++i) vector[i] = scalar_type_(distribution(generator));
        std::sort(vector.values_data(), vector.values_data() + count);
        unique_end = std::unique(vector.values_data(), vector.values_data() + count);
        unique_count = static_cast<std::size_t>(unique_end - vector.values_data());
    }
}

/**
 *  @brief Test set intersection (unified template for u16/u32 index types).
 */
template <typename index_type_>
error_stats_t test_intersect(typename index_type_::sparse_intersect_kernel_t kernel) {
    using index_t = index_type_;
    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    std::size_t dim = sparse_dimensions;
    auto a = make_vector<index_t>(dim), b = make_vector<index_t>(dim);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_sorted_unique(generator, a, index_t(dim * 4));
        fill_sorted_unique(generator, b, index_t(dim * 4));

        nk_size_t count;
        kernel(a.raw_values_data(), b.raw_values_data(), dim, dim, nullptr, &count);

        nk_size_t ref;
        nk::sparse_intersect<index_t, nk::no_simd_k>(a.values_data(), b.values_data(), dim, dim, &ref);
        stats.accumulate(count, ref);
    }
    return stats;
}

/**
 *  @brief Test sparse dot product (unified template, parameterized by weight type).
 *
 *  Dispatch is by weight type (matching numkong.h dispatch tables):
 *  - bf16_t weights -> u16_t indices
 *  - f32_t weights -> u32_t indices
 */
template <typename weight_type_>
error_stats_t test_sparse_dot(typename weight_type_::sparse_dot_kernel_t kernel) {
    using weight_t = weight_type_;
    using index_t = typename weight_t::sparse_dot_index_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    std::size_t dim = sparse_dimensions;
    auto a_idx = make_vector<index_t>(dim), b_idx = make_vector<index_t>(dim);
    auto a_weights = make_vector<weight_t>(dim), b_weights = make_vector<weight_t>(dim);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_sorted_unique(generator, a_idx, index_t(dim * 4));
        fill_sorted_unique(generator, b_idx, index_t(dim * 4));
        fill_random(generator, a_weights);
        fill_random(generator, b_weights);

        typename weight_t::dot_result_t result;
        kernel(a_idx.raw_values_data(), b_idx.raw_values_data(), a_weights.raw_values_data(),
               b_weights.raw_values_data(), dim, dim, &result.raw_);

        f118_t ref;
        nk::sparse_dot<index_t, weight_t, f118_t, nk::no_simd_k>(
            a_idx.values_data(), b_idx.values_data(), a_weights.values_data(), b_weights.values_data(), dim, dim, &ref);
        stats.accumulate(result, ref);
    }
    return stats;
}

void test_sparse() {
    std::puts("");
    std::printf("Sparse Operations:\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("sparse_intersect_u16", test_intersect<u16_t>, nk_sparse_intersect_u16);
    run_if_matches("sparse_intersect_u32", test_intersect<u32_t>, nk_sparse_intersect_u32);
    run_if_matches("sparse_intersect_u64", test_intersect<u64_t>, nk_sparse_intersect_u64);
    run_if_matches("sparse_dot_u32f32", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32);
    run_if_matches("sparse_dot_u16bf16", test_sparse_dot<bf16_t>, nk_sparse_dot_u16bf16);
#else

#if NK_TARGET_NEON
    run_if_matches("sparse_intersect_u16_neon", test_intersect<u16_t>, nk_sparse_intersect_u16_neon);
    run_if_matches("sparse_intersect_u32_neon", test_intersect<u32_t>, nk_sparse_intersect_u32_neon);
    run_if_matches("sparse_intersect_u64_neon", test_intersect<u64_t>, nk_sparse_intersect_u64_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_SVE
    run_if_matches("sparse_intersect_u16_sve2", test_intersect<u16_t>, nk_sparse_intersect_u16_sve2);
    run_if_matches("sparse_intersect_u32_sve2", test_intersect<u32_t>, nk_sparse_intersect_u32_sve2);
    run_if_matches("sparse_intersect_u64_sve2", test_intersect<u64_t>, nk_sparse_intersect_u64_sve2);
    run_if_matches("sparse_dot_u32f32_sve2", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32_sve2);
    run_if_matches("sparse_dot_u16bf16_sve2", test_sparse_dot<bf16_t>, nk_sparse_dot_u16bf16_sve2);
#endif // NK_TARGET_SVE

#if NK_TARGET_ICELAKE
    run_if_matches("sparse_intersect_u16_icelake", test_intersect<u16_t>, nk_sparse_intersect_u16_icelake);
    run_if_matches("sparse_intersect_u32_icelake", test_intersect<u32_t>, nk_sparse_intersect_u32_icelake);
    run_if_matches("sparse_intersect_u64_icelake", test_intersect<u64_t>, nk_sparse_intersect_u64_icelake);
    run_if_matches("sparse_dot_u32f32_icelake", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32_icelake);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_TURIN
    run_if_matches("sparse_intersect_u16_turin", test_intersect<u16_t>, nk_sparse_intersect_u16_turin);
    run_if_matches("sparse_intersect_u32_turin", test_intersect<u32_t>, nk_sparse_intersect_u32_turin);
    run_if_matches("sparse_intersect_u64_turin", test_intersect<u64_t>, nk_sparse_intersect_u64_turin);
    run_if_matches("sparse_dot_u32f32_turin", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32_turin);
    run_if_matches("sparse_dot_u16bf16_turin", test_sparse_dot<bf16_t>, nk_sparse_dot_u16bf16_turin);
#endif // NK_TARGET_TURIN

    // Serial always runs - baseline test
    run_if_matches("sparse_intersect_u16_serial", test_intersect<u16_t>, nk_sparse_intersect_u16_serial);
    run_if_matches("sparse_intersect_u32_serial", test_intersect<u32_t>, nk_sparse_intersect_u32_serial);
    run_if_matches("sparse_intersect_u64_serial", test_intersect<u64_t>, nk_sparse_intersect_u64_serial);
    run_if_matches("sparse_dot_u32f32_serial", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32_serial);
    run_if_matches("sparse_dot_u16bf16_serial", test_sparse_dot<bf16_t>, nk_sparse_dot_u16bf16_serial);

#endif // NK_DYNAMIC_DISPATCH
}
