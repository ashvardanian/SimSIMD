/**
 *  @brief Batch operation benchmarks - BLAS/MKL comparisons.
 *  @file bench/bench_cross_blas.cpp
 *  @author Ash Vardanian
 *  @date January 14, 2025
 */

#include <cstring> // std::memcpy
#include <vector>  // std::vector

#include "bench.hpp"

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

struct identity_init {
    template <typename scalar_type_>
    scalar_type_ operator()(scalar_type_ v) const noexcept {
        return v;
    }
};

template <typename input_type_, typename input_b_type_ = input_type_, typename output_type_ = input_type_,
          typename init_first_type_ = identity_init, typename init_second_type_ = identity_init, typename kernel_type_>
void measure_dots_unpacked(bm::State &state, std::size_t m, std::size_t n, std::size_t k, kernel_type_ kernel,
                           init_first_type_ init_first = {}, init_second_type_ init_second = {}) {
    std::vector<input_type_> matrix_a(m * k);
    std::vector<input_b_type_> matrix_b(n * k);
    std::vector<output_type_> matrix_c(m * n);
    auto generator = make_random_engine();
    nk::fill_uniform(generator, matrix_a.data(), matrix_a.size());
    nk::fill_uniform(generator, matrix_b.data(), matrix_b.size());

    std::size_t iterations = 0;
    for (auto _ : state) {
        bm::DoNotOptimize(matrix_c.data());
        kernel(matrix_a.data(), matrix_b.data(), matrix_c.data(), m, n, k);
        ++iterations;
    }
    state.counters["scalar-ops"] = bm::Counter(iterations * 2.0 * m * n * k, bm::Counter::kIsRate);
}

void measure_dots_f32_with_blas(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<float>(state, m, n, k,
                                 [](float *a, float *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
                                     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m),
                                                 static_cast<int>(n), static_cast<int>(k), 1.0f, a, static_cast<int>(k),
                                                 b, static_cast<int>(k), 0.0f, c, static_cast<int>(n));
                                 });
}

void measure_dots_f64_with_blas(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<double>(state, m, n, k,
                                  [](double *a, double *b, double *c, std::size_t m, std::size_t n, std::size_t k) {
                                      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m),
                                                  static_cast<int>(n), static_cast<int>(k), 1.0, a, static_cast<int>(k),
                                                  b, static_cast<int>(k), 0.0, c, static_cast<int>(n));
                                  });
}

template <typename input_type_, typename output_type_ = input_type_, typename init_type_ = identity_init,
          typename kernel_type_>
void measure_dots_symmetric_unpacked(bm::State &state, std::size_t n, std::size_t k, kernel_type_ kernel,
                                     init_type_ init = {}) {
    std::vector<input_type_> matrix_a(n * k);
    std::vector<output_type_> matrix_c(n * n);
    auto generator = make_random_engine();
    nk::fill_uniform(generator, matrix_a.data(), matrix_a.size());

    std::size_t iterations = 0;
    for (auto _ : state) {
        bm::DoNotOptimize(matrix_c.data());
        kernel(matrix_a.data(), matrix_c.data(), n, k);
        ++iterations;
    }
    state.counters["scalar-ops"] = bm::Counter(iterations * n * (n + 1) * k, bm::Counter::kIsRate);
}

void measure_dots_symmetric_f32_with_blas(bm::State &state, std::size_t n, std::size_t k) {
    measure_dots_symmetric_unpacked<float>(state, n, k, [](float *a, float *c, std::size_t n, std::size_t k) {
        cblas_ssyrk(CblasRowMajor, CblasUpper, CblasNoTrans, static_cast<int>(n), static_cast<int>(k), 1.0f, a,
                    static_cast<int>(k), 0.0f, c, static_cast<int>(n));
    });
}

void measure_dots_symmetric_f64_with_blas(bm::State &state, std::size_t n, std::size_t k) {
    measure_dots_symmetric_unpacked<double>(state, n, k, [](double *a, double *c, std::size_t n, std::size_t k) {
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, static_cast<int>(n), static_cast<int>(k), 1.0, a,
                    static_cast<int>(k), 0.0, c, static_cast<int>(n));
    });
}

#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

#if NK_COMPARE_TO_MKL

void measure_dots_f32_with_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<float>(state, m, n, k,
                                 [](float *a, float *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
                                     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n,
                                                 (MKL_INT)k, 1.0f, a, (MKL_INT)k, b, (MKL_INT)k, 0.0f, c, (MKL_INT)n);
                                 });
}

void measure_dots_bf16_with_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<MKL_BF16, MKL_BF16, float>(
        state, m, n, k,
        [](MKL_BF16 *a, MKL_BF16 *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n, (MKL_INT)k, 1.0f, a,
                                   (MKL_INT)k, b, (MKL_INT)k, 0.0f, c, (MKL_INT)n);
        },
        [](float val) -> MKL_BF16 {
            nk_bf16_t result;
            nk_f32_to_bf16(&val, &result);
            MKL_BF16 mkl_result;
            std::memcpy(&mkl_result, &result, sizeof(mkl_result));
            return mkl_result;
        },
        [](float val) -> MKL_BF16 {
            nk_bf16_t result;
            nk_f32_to_bf16(&val, &result);
            MKL_BF16 mkl_result;
            std::memcpy(&mkl_result, &result, sizeof(mkl_result));
            return mkl_result;
        });
}

void measure_dots_f16_with_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<MKL_F16, MKL_F16, float>(
        state, m, n, k,
        [](MKL_F16 *a, MKL_F16 *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
            cblas_gemm_f16f16f32(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n, (MKL_INT)k, 1.0f, a,
                                 (MKL_INT)k, b, (MKL_INT)k, 0.0f, c, (MKL_INT)n);
        },
        [](float val) -> MKL_F16 {
            nk_f16_t result;
            nk_f32_to_f16(&val, &result);
            MKL_F16 mkl_result;
            std::memcpy(&mkl_result, &result, sizeof(mkl_result));
            return mkl_result;
        },
        [](float val) -> MKL_F16 {
            nk_f16_t result;
            nk_f32_to_f16(&val, &result);
            MKL_F16 mkl_result;
            std::memcpy(&mkl_result, &result, sizeof(mkl_result));
            return mkl_result;
        });
}

void measure_dots_f64_with_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<double>(state, m, n, k,
                                  [](double *a, double *b, double *c, std::size_t m, std::size_t n, std::size_t k) {
                                      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n,
                                                  (MKL_INT)k, 1.0, a, (MKL_INT)k, b, (MKL_INT)k, 0.0, c, (MKL_INT)n);
                                  });
}

void measure_dots_u8i8i32_with_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<std::uint8_t, std::int8_t, std::int32_t>(
        state, m, n, k,
        [](std::uint8_t *a, std::int8_t *b, std::int32_t *c, std::size_t m, std::size_t n, std::size_t k) {
            MKL_INT32 c_offset = 0;
            cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, (MKL_INT)m, (MKL_INT)n,
                               (MKL_INT)k, 1.0f, a, (MKL_INT)k, 0, b, (MKL_INT)k, 0, 0.0f, c, (MKL_INT)n, &c_offset);
        });
}

void measure_dots_i16i16i32_with_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<std::int16_t, std::int16_t, std::int32_t>(
        state, m, n, k,
        [](std::int16_t *a, std::int16_t *b, std::int32_t *c, std::size_t m, std::size_t n, std::size_t k) {
            MKL_INT32 c_offset = 0;
            cblas_gemm_s16s16s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, (MKL_INT)m, (MKL_INT)n,
                                 (MKL_INT)k, 1.0f, a, (MKL_INT)k, 0, b, (MKL_INT)k, 0, 0.0f, c, (MKL_INT)n, &c_offset);
        });
}

#endif // NK_COMPARE_TO_MKL

void bench_cross_blas() {

    std::string syrk_dims = std::to_string(matrix_height) + "x" + std::to_string(matrix_depth);
    std::string gemm_dims = std::to_string(matrix_height) + "x" + std::to_string(matrix_width) + "x" +
                            std::to_string(matrix_depth);

    nk_unused_(syrk_dims);
    nk_unused_(gemm_dims);

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE
    // BLAS GEMM baselines for matmul comparison (same layout as NumKong: A x B^T)
    bm::RegisterBenchmark(("dots_f32_with_blas<" + gemm_dims + ">").c_str(), measure_dots_f32_with_blas, matrix_height,
                          matrix_width, matrix_depth);
    bm::RegisterBenchmark(("dots_f64_with_blas<" + gemm_dims + ">").c_str(), measure_dots_f64_with_blas, matrix_height,
                          matrix_width, matrix_depth);

    // BLAS SYRK baselines for symmetric operations (correct operation for dots_symmetric: A x A^T)
    bm::RegisterBenchmark(("dots_symmetric_f32_with_blas<" + syrk_dims + ">").c_str(),
                          measure_dots_symmetric_f32_with_blas, matrix_height, matrix_depth);
    bm::RegisterBenchmark(("dots_symmetric_f64_with_blas<" + syrk_dims + ">").c_str(),
                          measure_dots_symmetric_f64_with_blas, matrix_height, matrix_depth);
#endif

#if NK_COMPARE_TO_MKL
    bm::RegisterBenchmark(("dots_f32_with_mkl<" + gemm_dims + ">").c_str(), measure_dots_f32_with_mkl, matrix_height,
                          matrix_width, matrix_depth);
    bm::RegisterBenchmark(("dots_bf16_with_mkl<" + gemm_dims + ">").c_str(), measure_dots_bf16_with_mkl, matrix_height,
                          matrix_width, matrix_depth);
    bm::RegisterBenchmark(("dots_f16_with_mkl<" + gemm_dims + ">").c_str(), measure_dots_f16_with_mkl, matrix_height,
                          matrix_width, matrix_depth);
    bm::RegisterBenchmark(("dots_f64_with_mkl<" + gemm_dims + ">").c_str(), measure_dots_f64_with_mkl, matrix_height,
                          matrix_width, matrix_depth);
    bm::RegisterBenchmark(("dots_u8i8i32_with_mkl<" + gemm_dims + ">").c_str(), measure_dots_u8i8i32_with_mkl,
                          matrix_height, matrix_width, matrix_depth);
    bm::RegisterBenchmark(("dots_i16i16i32_with_mkl<" + gemm_dims + ">").c_str(), measure_dots_i16i16i32_with_mkl,
                          matrix_height, matrix_width, matrix_depth);
#endif
}
