/**
 *  @brief Mesh alignment benchmarks (RMSD, Kabsch, Umeyama).
 *  @file bench/bench_mesh.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 */

#include "numkong/mesh.h"

#include "bench.hpp"

/**
 *  @brief Measures the performance of a @b mesh kernel function (RMSD/Kabsch/Umeyama) using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param points_count The number of 3D points in each point cloud.
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void measure_mesh(bm::State &state, kernel_type_ kernel, std::size_t points_count) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename nk::type_for<output_dtype_>::type;
    using raw_output_t = typename output_t::raw_t;
    using input_vector_t = nk::vector<input_t>;

    // Preallocate point clouds: each contains points_count 3D points stored as [x0,y0,z0,x1,y1,z1,...]
    constexpr std::size_t clouds_count = 1024;
    std::vector<input_vector_t> first_clouds(clouds_count), second_clouds(clouds_count);
    auto generator = make_random_engine();
    for (std::size_t index = 0; index != clouds_count; ++index) {
        first_clouds[index] = make_vector<input_t>(points_count * 3);
        second_clouds[index] = make_vector<input_t>(points_count * 3);
        nk::fill_uniform(generator, first_clouds[index].values_data(), first_clouds[index].size_values());
        nk::fill_uniform(generator, second_clouds[index].values_data(), second_clouds[index].size_values());
    }

    // Benchmark loop
    std::size_t iterations = 0;
    for (auto _ : state) {
        output_t result, scale;
        raw_output_t first_centroid[3], second_centroid[3], rotation[9];
        std::size_t const index = iterations & (clouds_count - 1);
        kernel(first_clouds[index].raw_values_data(), second_clouds[index].raw_values_data(), points_count,
               first_centroid, second_centroid, rotation, &scale.raw_, &result.raw_);
        bm::DoNotOptimize(result);
        iterations++;
    }

    state.counters["bytes"] = bm::Counter(iterations * first_clouds[0].size_bytes() * 2, bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void mesh_(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(mesh_points) + "pts>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_mesh<input_dtype_, output_dtype_, kernel_type_ *>, kernel,
                          mesh_points);
}

void bench_mesh() {
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;

#if NK_TARGET_NEON
    mesh_<f32_k, f32_k>("rmsd_f32_neon", nk_rmsd_f32_neon);
    mesh_<f32_k, f32_k>("kabsch_f32_neon", nk_kabsch_f32_neon);
    mesh_<f32_k, f32_k>("umeyama_f32_neon", nk_umeyama_f32_neon);
    mesh_<f64_k, f64_k>("rmsd_f64_neon", nk_rmsd_f64_neon);
    mesh_<f64_k, f64_k>("kabsch_f64_neon", nk_kabsch_f64_neon);
    mesh_<f64_k, f64_k>("umeyama_f64_neon", nk_umeyama_f64_neon);
#endif


#if NK_TARGET_HASWELL
    mesh_<f32_k, f32_k>("rmsd_f32_haswell", nk_rmsd_f32_haswell);
    mesh_<f32_k, f32_k>("kabsch_f32_haswell", nk_kabsch_f32_haswell);
    mesh_<f32_k, f32_k>("umeyama_f32_haswell", nk_umeyama_f32_haswell);
    mesh_<f64_k, f64_k>("rmsd_f64_haswell", nk_rmsd_f64_haswell);
    mesh_<f64_k, f64_k>("kabsch_f64_haswell", nk_kabsch_f64_haswell);
    mesh_<f64_k, f64_k>("umeyama_f64_haswell", nk_umeyama_f64_haswell);
    mesh_<f16_k, f32_k>("rmsd_f16_haswell", nk_rmsd_f16_haswell);
    mesh_<f16_k, f32_k>("kabsch_f16_haswell", nk_kabsch_f16_haswell);
    mesh_<f16_k, f32_k>("umeyama_f16_haswell", nk_umeyama_f16_haswell);
    mesh_<bf16_k, f32_k>("rmsd_bf16_haswell", nk_rmsd_bf16_haswell);
    mesh_<bf16_k, f32_k>("kabsch_bf16_haswell", nk_kabsch_bf16_haswell);
    mesh_<bf16_k, f32_k>("umeyama_bf16_haswell", nk_umeyama_bf16_haswell);
#endif

#if NK_TARGET_SKYLAKE
    mesh_<f32_k, f32_k>("rmsd_f32_skylake", nk_rmsd_f32_skylake);
    mesh_<f32_k, f32_k>("kabsch_f32_skylake", nk_kabsch_f32_skylake);
#endif

    // Serial fallbacks
    mesh_<f32_k, f32_k>("rmsd_f32_serial", nk_rmsd_f32_serial);
    mesh_<f32_k, f32_k>("kabsch_f32_serial", nk_kabsch_f32_serial);
    mesh_<f32_k, f32_k>("umeyama_f32_serial", nk_umeyama_f32_serial);
    mesh_<f64_k, f64_k>("rmsd_f64_serial", nk_rmsd_f64_serial);
    mesh_<f64_k, f64_k>("kabsch_f64_serial", nk_kabsch_f64_serial);
    mesh_<f64_k, f64_k>("umeyama_f64_serial", nk_umeyama_f64_serial);
}
