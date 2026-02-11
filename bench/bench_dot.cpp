/**
 *  @brief Dot product benchmarks.
 *  @file bench/bench_dot.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 */

#include <complex> // std::complex

#include "numkong/dot.h"

#include "bench.hpp"

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

void dot_f32_with_blas(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    *result = cblas_sdot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f64_with_blas(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    *result = cblas_ddot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f32c_with_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_cdotu_sub(static_cast<int>(n), reinterpret_cast<std::complex<float> const *>(a), 1,
                    reinterpret_cast<std::complex<float> const *>(b), 1,
                    reinterpret_cast<std::complex<float> *>(result));
#else
    cblas_cdotu_sub(static_cast<int>(n), reinterpret_cast<nk_f32_t const *>(a), 1,
                    reinterpret_cast<nk_f32_t const *>(b), 1, reinterpret_cast<nk_f32_t *>(result));
#endif
}

void dot_f64c_with_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_zdotu_sub(static_cast<int>(n), reinterpret_cast<std::complex<double> const *>(a), 1,
                    reinterpret_cast<std::complex<double> const *>(b), 1,
                    reinterpret_cast<std::complex<double> *>(result));
#else
    cblas_zdotu_sub(static_cast<int>(n), reinterpret_cast<nk_f64_t const *>(a), 1,
                    reinterpret_cast<nk_f64_t const *>(b), 1, reinterpret_cast<nk_f64_t *>(result));
#endif
}

void vdot_f32c_with_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_cdotc_sub(static_cast<int>(n), reinterpret_cast<std::complex<float> const *>(a), 1,
                    reinterpret_cast<std::complex<float> const *>(b), 1,
                    reinterpret_cast<std::complex<float> *>(result));
#else
    cblas_cdotc_sub(static_cast<int>(n), reinterpret_cast<nk_f32_t const *>(a), 1,
                    reinterpret_cast<nk_f32_t const *>(b), 1, reinterpret_cast<nk_f32_t *>(result));
#endif
}

void vdot_f64c_with_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_zdotc_sub(static_cast<int>(n), reinterpret_cast<std::complex<double> const *>(a), 1,
                    reinterpret_cast<std::complex<double> const *>(b), 1,
                    reinterpret_cast<std::complex<double> *>(result));
#else
    cblas_zdotc_sub(static_cast<int>(n), reinterpret_cast<nk_f64_t const *>(a), 1,
                    reinterpret_cast<nk_f64_t const *>(b), 1, reinterpret_cast<nk_f64_t *>(result));
#endif
}

#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

void bench_dot() {
    constexpr nk_dtype_t u1_k = nk_u1_k;
    constexpr nk_dtype_t i4_k = nk_i4_k;
    constexpr nk_dtype_t u4_k = nk_u4_k;
    constexpr nk_dtype_t i8_k = nk_i8_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t i32_k = nk_i32_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t e4m3_k = nk_e4m3_k;
    constexpr nk_dtype_t e5m2_k = nk_e5m2_k;
    constexpr nk_dtype_t e2m3_k = nk_e2m3_k;
    constexpr nk_dtype_t e3m2_k = nk_e3m2_k;
    constexpr nk_dtype_t f64c_k = nk_f64c_k;
    constexpr nk_dtype_t f32c_k = nk_f32c_k;
    constexpr nk_dtype_t f16c_k = nk_f16c_k;
    constexpr nk_dtype_t bf16c_k = nk_bf16c_k;

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE
    dense_<f32_k, f32_k>("dot_f32_with_blas", dot_f32_with_blas);
    dense_<f64_k, f64_k>("dot_f64_with_blas", dot_f64_with_blas);
    dense_<f32c_k, f32c_k>("dot_f32c_with_blas", dot_f32c_with_blas);
    dense_<f64c_k, f64c_k>("dot_f64c_with_blas", dot_f64c_with_blas);
    dense_<f32c_k, f32c_k>("vdot_f32c_with_blas", vdot_f32c_with_blas);
    dense_<f64c_k, f64c_k>("vdot_f64c_with_blas", vdot_f64c_with_blas);
#endif

#if NK_TARGET_NEON
    dense_<f32_k, f32_k>("dot_f32_neon", nk_dot_f32_neon);
    dense_<f32c_k, f32c_k>("dot_f32c_neon", nk_dot_f32c_neon);
    dense_<f32c_k, f32c_k>("vdot_f32c_neon", nk_vdot_f32c_neon);
    dense_<bf16_k, f32_k>("dot_bf16_neon", nk_dot_bf16_neon);
    dense_<e4m3_k, f32_k>("dot_e4m3_neon", nk_dot_e4m3_neon);
    dense_<e5m2_k, f32_k>("dot_e5m2_neon", nk_dot_e5m2_neon);
    dense_<e2m3_k, f32_k>("dot_e2m3_neon", nk_dot_e2m3_neon);
    dense_<e3m2_k, f32_k>("dot_e3m2_neon", nk_dot_e3m2_neon);
#endif

#if NK_TARGET_NEONSDOT
    dense_<i8_k, i32_k>("dot_i8_neonsdot", nk_dot_i8_neonsdot);
    dense_<u8_k, u32_k>("dot_u8_neonsdot", nk_dot_u8_neonsdot);
    dense_<i4_k, i32_k>("dot_i4_neonsdot", nk_dot_i4_neonsdot);
    dense_<u4_k, u32_k>("dot_u4_neonsdot", nk_dot_u4_neonsdot);
    dense_<e2m3_k, f32_k>("dot_e2m3_neonsdot", nk_dot_e2m3_neonsdot);
    dense_<e3m2_k, f32_k>("dot_e3m2_neonsdot", nk_dot_e3m2_neonsdot);
#endif

#if NK_TARGET_NEONHALF
    dense_<f16c_k, f32c_k>("dot_f16c_neonhalf", nk_dot_f16c_neonhalf);
    dense_<f16c_k, f32c_k>("vdot_f16c_neonhalf", nk_vdot_f16c_neonhalf);
    dense_<f16_k, f32_k>("dot_f16_neonhalf", nk_dot_f16_neonhalf);
#endif

#if NK_TARGET_NEONFHM
    dense_<f16_k, f32_k>("dot_f16_neonfhm", nk_dot_f16_neonfhm);
    dense_<e2m3_k, f32_k>("dot_e2m3_neonfhm", nk_dot_e2m3_neonfhm);
    dense_<e3m2_k, f32_k>("dot_e3m2_neonfhm", nk_dot_e3m2_neonfhm);
    dense_<e4m3_k, f32_k>("dot_e4m3_neonfhm", nk_dot_e4m3_neonfhm);
    dense_<e5m2_k, f32_k>("dot_e5m2_neonfhm", nk_dot_e5m2_neonfhm);
#endif

#if NK_TARGET_NEONBFDOT
    dense_<bf16c_k, f32c_k>("dot_bf16c_neonbfdot", nk_dot_bf16c_neonbfdot);
    dense_<bf16c_k, f32c_k>("vdot_bf16c_neonbfdot", nk_vdot_bf16c_neonbfdot);
    dense_<bf16_k, f32_k>("dot_bf16_neonbfdot", nk_dot_bf16_neonbfdot);
    dense_<e4m3_k, f32_k>("dot_e4m3_neonbfdot", nk_dot_e4m3_neonbfdot);
    dense_<e5m2_k, f32_k>("dot_e5m2_neonbfdot", nk_dot_e5m2_neonbfdot);
#endif

#if NK_TARGET_SVE
    dense_<f32_k, f32_k>("dot_f32_sve", nk_dot_f32_sve);
    dense_<f64_k, f64_k>("dot_f64_sve", nk_dot_f64_sve);
    dense_<f32c_k, f32c_k>("dot_f32c_sve", nk_dot_f32c_sve);
    dense_<f32c_k, f32c_k>("vdot_f32c_sve", nk_vdot_f32c_sve);
    dense_<f64c_k, f64c_k>("dot_f64c_sve", nk_dot_f64c_sve);
    dense_<f64c_k, f64c_k>("vdot_f64c_sve", nk_vdot_f64c_sve);
#endif

#if NK_TARGET_SVEHALF
    dense_<f16_k, f32_k>("dot_f16_svehalf", nk_dot_f16_svehalf);
    dense_<f16c_k, f32c_k>("dot_f16c_svehalf", nk_dot_f16c_svehalf);
    dense_<f16c_k, f32c_k>("vdot_f16c_svehalf", nk_vdot_f16c_svehalf);
#endif

#if NK_TARGET_HASWELL
    dense_<f16_k, f32_k>("dot_f16_haswell", nk_dot_f16_haswell);
    dense_<bf16_k, f32_k>("dot_bf16_haswell", nk_dot_bf16_haswell);
    dense_<e4m3_k, f32_k>("dot_e4m3_haswell", nk_dot_e4m3_haswell);
    dense_<e5m2_k, f32_k>("dot_e5m2_haswell", nk_dot_e5m2_haswell);
    dense_<e2m3_k, f32_k>("dot_e2m3_haswell", nk_dot_e2m3_haswell);
    dense_<e3m2_k, f32_k>("dot_e3m2_haswell", nk_dot_e3m2_haswell);
    dense_<i8_k, i32_k>("dot_i8_haswell", nk_dot_i8_haswell);
    dense_<u8_k, u32_k>("dot_u8_haswell", nk_dot_u8_haswell);
    dense_<f16c_k, f32c_k>("dot_f16c_haswell", nk_dot_f16c_haswell);
    dense_<f16c_k, f32c_k>("vdot_f16c_haswell", nk_vdot_f16c_haswell);
    dense_<f32c_k, f32c_k>("dot_f32c_haswell", nk_dot_f32c_haswell);
    dense_<f32c_k, f32c_k>("vdot_f32c_haswell", nk_vdot_f32c_haswell);
    dense_<bf16c_k, f32c_k>("dot_bf16c_haswell", nk_dot_bf16c_haswell);
    dense_<bf16c_k, f32c_k>("vdot_bf16c_haswell", nk_vdot_bf16c_haswell);
    dense_<i4_k, i32_k>("dot_i4_haswell", nk_dot_i4_haswell);
    dense_<u4_k, u32_k>("dot_u4_haswell", nk_dot_u4_haswell);
#endif

#if NK_TARGET_SKYLAKE
    dense_<f32_k, f32_k>("dot_f32_skylake", nk_dot_f32_skylake);
    dense_<f64_k, f64_k>("dot_f64_skylake", nk_dot_f64_skylake);
    dense_<bf16_k, f32_k>("dot_bf16_skylake", nk_dot_bf16_skylake);
    dense_<f16_k, f32_k>("dot_f16_skylake", nk_dot_f16_skylake);
    dense_<e4m3_k, f32_k>("dot_e4m3_skylake", nk_dot_e4m3_skylake);
    dense_<e5m2_k, f32_k>("dot_e5m2_skylake", nk_dot_e5m2_skylake);
    dense_<e2m3_k, f32_k>("dot_e2m3_skylake", nk_dot_e2m3_skylake);
    dense_<e3m2_k, f32_k>("dot_e3m2_skylake", nk_dot_e3m2_skylake);
    dense_<i8_k, i32_k>("dot_i8_skylake", nk_dot_i8_skylake);
    dense_<u8_k, u32_k>("dot_u8_skylake", nk_dot_u8_skylake);
    dense_<f32c_k, f32c_k>("dot_f32c_skylake", nk_dot_f32c_skylake);
    dense_<f32c_k, f32c_k>("vdot_f32c_skylake", nk_vdot_f32c_skylake);
    dense_<f64c_k, f64c_k>("dot_f64c_skylake", nk_dot_f64c_skylake);
    dense_<f64c_k, f64c_k>("vdot_f64c_skylake", nk_vdot_f64c_skylake);
#endif

#if NK_TARGET_ICELAKE
    dense_<i8_k, i32_k>("dot_i8_icelake", nk_dot_i8_icelake);
    dense_<u8_k, u32_k>("dot_u8_icelake", nk_dot_u8_icelake);
    dense_<i4_k, i32_k>("dot_i4_icelake", nk_dot_i4_icelake);
    dense_<u4_k, u32_k>("dot_u4_icelake", nk_dot_u4_icelake);
    dense_<e2m3_k, f32_k>("dot_e2m3_icelake", nk_dot_e2m3_icelake);
    dense_<e3m2_k, f32_k>("dot_e3m2_icelake", nk_dot_e3m2_icelake);
#endif

#if NK_TARGET_GENOA
    dense_<bf16_k, f32_k>("dot_bf16_genoa", nk_dot_bf16_genoa);
    dense_<bf16c_k, f32c_k>("dot_bf16c_genoa", nk_dot_bf16c_genoa);
    dense_<bf16c_k, f32c_k>("vdot_bf16c_genoa", nk_vdot_bf16c_genoa);
    dense_<e4m3_k, f32_k>("dot_e4m3_genoa", nk_dot_e4m3_genoa);
    dense_<e5m2_k, f32_k>("dot_e5m2_genoa", nk_dot_e5m2_genoa);
    dense_<e2m3_k, f32_k>("dot_e2m3_genoa", nk_dot_e2m3_genoa);
    dense_<e3m2_k, f32_k>("dot_e3m2_genoa", nk_dot_e3m2_genoa);
#endif

#if NK_TARGET_SAPPHIRE
    dense_<e3m2_k, f32_k>("dot_e3m2_sapphire", nk_dot_e3m2_sapphire);
#endif

#if NK_TARGET_RVV
    dense_<i8_k, i32_k>("dot_i8_rvv", nk_dot_i8_rvv);
    dense_<u8_k, u32_k>("dot_u8_rvv", nk_dot_u8_rvv);
    dense_<f32_k, f32_k>("dot_f32_rvv", nk_dot_f32_rvv);
    dense_<f64_k, f64_k>("dot_f64_rvv", nk_dot_f64_rvv);
#endif

#if NK_TARGET_V128RELAXED
    dense_<f32_k, f32_k>("dot_f32_v128relaxed", nk_dot_f32_v128relaxed);
    dense_<f64_k, f64_k>("dot_f64_v128relaxed", nk_dot_f64_v128relaxed);
    dense_<f16_k, f32_k>("dot_f16_v128relaxed", nk_dot_f16_v128relaxed);
    dense_<bf16_k, f32_k>("dot_bf16_v128relaxed", nk_dot_bf16_v128relaxed);
    dense_<i8_k, i32_k>("dot_i8_v128relaxed", nk_dot_i8_v128relaxed);
    dense_<u8_k, u32_k>("dot_u8_v128relaxed", nk_dot_u8_v128relaxed);
    dense_<e2m3_k, f32_k>("dot_e2m3_v128relaxed", nk_dot_e2m3_v128relaxed);
    dense_<e3m2_k, f32_k>("dot_e3m2_v128relaxed", nk_dot_e3m2_v128relaxed);
#endif

    // Serial fallbacks
    dense_<bf16_k, f32_k>("dot_bf16_serial", nk_dot_bf16_serial);
    dense_<e4m3_k, f32_k>("dot_e4m3_serial", nk_dot_e4m3_serial);
    dense_<e5m2_k, f32_k>("dot_e5m2_serial", nk_dot_e5m2_serial);
    dense_<e2m3_k, f32_k>("dot_e2m3_serial", nk_dot_e2m3_serial);
    dense_<e3m2_k, f32_k>("dot_e3m2_serial", nk_dot_e3m2_serial);
    dense_<f16_k, f32_k>("dot_f16_serial", nk_dot_f16_serial);
    dense_<f32_k, f32_k>("dot_f32_serial", nk_dot_f32_serial);
    dense_<f64_k, f64_k>("dot_f64_serial", nk_dot_f64_serial);
    dense_<i8_k, i32_k>("dot_i8_serial", nk_dot_i8_serial);
    dense_<u8_k, u32_k>("dot_u8_serial", nk_dot_u8_serial);
    dense_<i4_k, i32_k>("dot_i4_serial", nk_dot_i4_serial);
    dense_<u4_k, u32_k>("dot_u4_serial", nk_dot_u4_serial);
    dense_<f64c_k, f64c_k>("dot_f64c_serial", nk_dot_f64c_serial);
    dense_<f32c_k, f32c_k>("dot_f32c_serial", nk_dot_f32c_serial);
    dense_<f16c_k, f32c_k>("dot_f16c_serial", nk_dot_f16c_serial);
    dense_<bf16c_k, f32c_k>("dot_bf16c_serial", nk_dot_bf16c_serial);
    dense_<f64c_k, f64c_k>("vdot_f64c_serial", nk_vdot_f64c_serial);
    dense_<f32c_k, f32c_k>("vdot_f32c_serial", nk_vdot_f32c_serial);
    dense_<f16c_k, f32c_k>("vdot_f16c_serial", nk_vdot_f16c_serial);
    dense_<bf16c_k, f32c_k>("vdot_bf16c_serial", nk_vdot_bf16c_serial);
}
