/**
 *  @brief Binary set operations tests (Hamming, Jaccard).
 *  @file test/test_set.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "test.hpp"
#include "numkong/set.hpp"

/**
 *  @brief Test Hamming distance for binary vectors.
 */
error_stats_t test_hamming(u1x8_t::hamming_kernel_t kernel) {
    using scalar_t = u1x8_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);

    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        u32_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), global_config.dense_dimensions, &result.raw_);

        u32_t reference;
        nk::hamming(a.values_data(), b.values_data(), global_config.dense_dimensions, &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

/**
 *  @brief Test Jaccard distance for binary vectors.
 */
error_stats_t test_jaccard(u1x8_t::jaccard_kernel_t kernel) {
    using scalar_t = u1x8_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);

    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        f32_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), global_config.dense_dimensions, &result.raw_);

        f32_t reference;
        nk::jaccard(a.values_data(), b.values_data(), global_config.dense_dimensions, &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

void test_set() {
    std::puts("");
    std::printf("Binary Distances:\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("hamming_u1", test_hamming, nk_hamming_u1);
    run_if_matches("jaccard_u1", test_jaccard, nk_jaccard_u1);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("hamming_u1_neon", test_hamming, nk_hamming_u1_neon);
    run_if_matches("jaccard_u1_neon", test_jaccard, nk_jaccard_u1_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
    run_if_matches("hamming_u1_haswell", test_hamming, nk_hamming_u1_haswell);
    run_if_matches("jaccard_u1_haswell", test_jaccard, nk_jaccard_u1_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_ICELAKE
    run_if_matches("hamming_u1_icelake", test_hamming, nk_hamming_u1_icelake);
    run_if_matches("jaccard_u1_icelake", test_jaccard, nk_jaccard_u1_icelake);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_RVV
    run_if_matches("hamming_u1_rvv", test_hamming, nk_hamming_u1_rvv);
    run_if_matches("jaccard_u1_rvv", test_jaccard, nk_jaccard_u1_rvv);
#endif // NK_TARGET_RVV

#if NK_TARGET_V128RELAXED
    run_if_matches("hamming_u1_v128relaxed", test_hamming, nk_hamming_u1_v128relaxed);
    run_if_matches("jaccard_u1_v128relaxed", test_jaccard, nk_jaccard_u1_v128relaxed);
#endif // NK_TARGET_V128RELAXED

    // Serial always runs - baseline test
    run_if_matches("hamming_u1_serial", test_hamming, nk_hamming_u1_serial);
    run_if_matches("jaccard_u1_serial", test_jaccard, nk_jaccard_u1_serial);

#endif // NK_DYNAMIC_DISPATCH
}
