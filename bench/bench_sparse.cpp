/**
 *  @brief Sparse operations benchmarks (sparse_intersect).
 *  @file bench/bench_sparse.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 */

#include <algorithm>     // std::sort
#include <limits>        // std::numeric_limits
#include <unordered_set> // std::unordered_set

#include "numkong/sparse.h"

#include "bench.hpp"

/**
 *  @brief Measures the performance of a @b sparse (set intersection) kernel function using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param first_size The number of elements in the first (smaller) set.
 *  @param second_size The number of elements in the second (larger) set.
 *  @param intersection_size The expected number of common elements between the sets.
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void measure_sparse(bm::State &state, kernel_type_ kernel, std::size_t first_size, std::size_t second_size,
                    std::size_t intersection_size) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using input_vector_t = nk::vector<input_t>;
    using scalar_t = typename input_t::raw_t;

    // Preallocate sorted unique set vectors
    std::size_t bytes_per_set = bench_dtype_bytes(input_dtype_, first_size + second_size);
    std::size_t const vectors_count = bench_input_count(bytes_per_set);
    std::vector<input_vector_t> first_vectors(vectors_count), second_vectors(vectors_count);
    auto generator = make_random_engine();
    std::uniform_int_distribution<scalar_t> distribution(0, std::numeric_limits<scalar_t>::max());

    // Generating sorted unique sets with controlled intersection is complex:
    // we generate intersection elements, then unique elements for each set.
    std::unordered_set<scalar_t> intersection_elements, unique_first, unique_second;
    intersection_elements.reserve(intersection_size);
    unique_first.reserve(first_size - intersection_size);
    unique_second.reserve(second_size - intersection_size);

    for (std::size_t index = 0; index != vectors_count; ++index) {
        first_vectors[index] = make_vector<input_t>(first_size);
        second_vectors[index] = make_vector<input_t>(second_size);

        // Step 1: Generate intersection elements
        intersection_elements.clear();
        while (intersection_elements.size() < intersection_size) intersection_elements.insert(distribution(generator));

        unique_first.clear();
        while (unique_first.size() < first_size - intersection_size) {
            scalar_t element = distribution(generator);
            if (intersection_elements.find(element) == intersection_elements.end()) unique_first.insert(element);
        }

        unique_second.clear();
        while (unique_second.size() < second_size - intersection_size) {
            scalar_t element = distribution(generator);
            if (intersection_elements.find(element) == intersection_elements.end() &&
                unique_first.find(element) == unique_first.end())
                unique_second.insert(element);
        }

        // Step 2: Merge and sort
        input_t *first_data = first_vectors[index].values_data();
        input_t *second_data = second_vectors[index].values_data();
        std::size_t offset = 0;
        for (scalar_t element : intersection_elements) first_data[offset++].raw_ = element;
        for (scalar_t element : unique_first) first_data[offset++].raw_ = element;
        offset = 0;
        for (scalar_t element : intersection_elements) second_data[offset++].raw_ = element;
        for (scalar_t element : unique_second) second_data[offset++].raw_ = element;
        std::sort(first_data, first_data + first_size, [](input_t a, input_t b) { return a.raw_ < b.raw_; });
        std::sort(second_data, second_data + second_size, [](input_t a, input_t b) { return a.raw_ < b.raw_; });
    }

    // Benchmark loop
    std::size_t iterations = 0;
    for (auto _ : state) {
        nk_size_t count;
        std::size_t const index = iterations & (vectors_count - 1);
        kernel(first_vectors[index].raw_values_data(), second_vectors[index].raw_values_data(), first_size, second_size,
               nullptr, &count);
        bm::DoNotOptimize(count);
        iterations++;
    }

    state.counters["bytes"] = bm::Counter(iterations * (first_vectors[0].size_bytes() + second_vectors[0].size_bytes()),
                                          bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void sparse_(std::string name, kernel_type_ *kernel) {
    std::size_t intersection_size = static_cast<std::size_t>(
        std::min(bench_config.sparse_first_length, bench_config.sparse_second_length) *
        bench_config.sparse_intersection_share);
    std::string bench_name = name + "<|A|=" + std::to_string(bench_config.sparse_first_length) +
                             ",|B|=" + std::to_string(bench_config.sparse_second_length) +
                             ",|A^B|=" + std::to_string(intersection_size) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_sparse<input_dtype_, output_dtype_, kernel_type_ *>, kernel,
                          bench_config.sparse_first_length, bench_config.sparse_second_length, intersection_size);
}

void bench_sparse() {
    constexpr nk_dtype_t u16_k = nk_u16_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t u64_k = nk_u64_k;

#if NK_TARGET_NEON
    sparse_<u16_k, u64_k>("sparse_intersect_u16_neon", nk_sparse_intersect_u16_neon);
    sparse_<u32_k, u64_k>("sparse_intersect_u32_neon", nk_sparse_intersect_u32_neon);
    sparse_<u64_k, u64_k>("sparse_intersect_u64_neon", nk_sparse_intersect_u64_neon);
#endif

#if NK_TARGET_SVE2
    sparse_<u16_k, u32_k>("sparse_intersect_u16_sve2", nk_sparse_intersect_u16_sve2);
    sparse_<u32_k, u32_k>("sparse_intersect_u32_sve2", nk_sparse_intersect_u32_sve2);
    sparse_<u64_k, u64_k>("sparse_intersect_u64_sve2", nk_sparse_intersect_u64_sve2);
#endif

#if NK_TARGET_ICELAKE
    sparse_<u16_k, u32_k>("sparse_intersect_u16_icelake", nk_sparse_intersect_u16_icelake);
    sparse_<u32_k, u32_k>("sparse_intersect_u32_icelake", nk_sparse_intersect_u32_icelake);
    sparse_<u64_k, u64_k>("sparse_intersect_u64_icelake", nk_sparse_intersect_u64_icelake);
#endif

#if NK_TARGET_TURIN
    sparse_<u16_k, u32_k>("sparse_intersect_u16_turin", nk_sparse_intersect_u16_turin);
    sparse_<u32_k, u32_k>("sparse_intersect_u32_turin", nk_sparse_intersect_u32_turin);
    sparse_<u64_k, u64_k>("sparse_intersect_u64_turin", nk_sparse_intersect_u64_turin);
#endif

    // Serial fallbacks
    sparse_<u16_k, u32_k>("sparse_intersect_u16_serial", nk_sparse_intersect_u16_serial);
    sparse_<u32_k, u32_k>("sparse_intersect_u32_serial", nk_sparse_intersect_u32_serial);
    sparse_<u64_k, u64_k>("sparse_intersect_u64_serial", nk_sparse_intersect_u64_serial);
}
