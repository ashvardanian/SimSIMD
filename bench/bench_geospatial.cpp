/**
 *  @brief Geospatial distance benchmarks (haversine, vincenty).
 *  @file bench/bench_geospatial.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 */

#include "numkong/geospatial.h"

#include "bench.hpp"

/**
 *  @brief Measures the performance of geospatial operations (Haversine/Vincenty) using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param coordinates_count The number of coordinate pairs to process.
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void measure_geospatial(bm::State &state, kernel_type_ kernel, std::size_t coordinates_count) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename nk::type_for<output_dtype_>::type;
    using input_vector_t = nk::vector<input_t>;
    using output_vector_t = nk::vector<output_t>;

    // Preallocate coordinate arrays: latitude1, longitude1, latitude2, longitude2
    constexpr std::size_t batches_count = 1024;
    std::vector<input_vector_t> latitudes_first(batches_count), longitudes_first(batches_count);
    std::vector<input_vector_t> latitudes_second(batches_count), longitudes_second(batches_count);
    auto generator = make_random_engine();
    for (std::size_t index = 0; index != batches_count; ++index) {
        latitudes_first[index] = make_vector<input_t>(coordinates_count);
        longitudes_first[index] = make_vector<input_t>(coordinates_count);
        latitudes_second[index] = make_vector<input_t>(coordinates_count);
        longitudes_second[index] = make_vector<input_t>(coordinates_count);
        nk::fill_coordinates(generator, latitudes_first[index].values_data(), longitudes_first[index].values_data(),
                             coordinates_count);
        nk::fill_coordinates(generator, latitudes_second[index].values_data(), longitudes_second[index].values_data(),
                             coordinates_count);
    }

    // Output distances buffer
    output_vector_t distances = make_vector<output_t>(coordinates_count);

    // Benchmark loop
    std::size_t iterations = 0;
    for (auto _ : state) {
        std::size_t const index = iterations & (batches_count - 1);
        kernel(latitudes_first[index].raw_values_data(), longitudes_first[index].raw_values_data(),
               latitudes_second[index].raw_values_data(), longitudes_second[index].raw_values_data(), coordinates_count,
               distances.raw_values_data());
        bm::ClobberMemory();
        iterations++;
    }

    std::size_t const bytes_per_call = latitudes_first[0].size_bytes() * 4; // 4 coordinate arrays
    state.counters["bytes"] = bm::Counter(iterations * bytes_per_call, bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations * coordinates_count, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void geospatial_(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_geospatial<input_dtype_, output_dtype_, kernel_type_ *>, kernel,
                          dense_dimensions)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

void bench_geospatial() {
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;

#if NK_TARGET_NEON
    geospatial_<f32_k, f32_k>("haversine_f32_neon", nk_haversine_f32_neon);
    geospatial_<f64_k, f64_k>("haversine_f64_neon", nk_haversine_f64_neon);
    geospatial_<f32_k, f32_k>("vincenty_f32_neon", nk_vincenty_f32_neon);
    geospatial_<f64_k, f64_k>("vincenty_f64_neon", nk_vincenty_f64_neon);
#endif

#if NK_TARGET_HASWELL
    geospatial_<f32_k, f32_k>("haversine_f32_haswell", nk_haversine_f32_haswell);
    geospatial_<f64_k, f64_k>("haversine_f64_haswell", nk_haversine_f64_haswell);
    geospatial_<f32_k, f32_k>("vincenty_f32_haswell", nk_vincenty_f32_haswell);
    geospatial_<f64_k, f64_k>("vincenty_f64_haswell", nk_vincenty_f64_haswell);
#endif

#if NK_TARGET_SKYLAKE
    geospatial_<f32_k, f32_k>("haversine_f32_skylake", nk_haversine_f32_skylake);
    geospatial_<f64_k, f64_k>("haversine_f64_skylake", nk_haversine_f64_skylake);
    geospatial_<f32_k, f32_k>("vincenty_f32_skylake", nk_vincenty_f32_skylake);
    geospatial_<f64_k, f64_k>("vincenty_f64_skylake", nk_vincenty_f64_skylake);
#endif

    // Serial fallbacks
    geospatial_<f32_k, f32_k>("haversine_f32_serial", nk_haversine_f32_serial);
    geospatial_<f64_k, f64_k>("haversine_f64_serial", nk_haversine_f64_serial);
    geospatial_<f32_k, f32_k>("vincenty_f32_serial", nk_vincenty_f32_serial);
    geospatial_<f64_k, f64_k>("vincenty_f64_serial", nk_vincenty_f64_serial);
}
