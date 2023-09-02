import platform

import simsimd as simd


def test_non_null():
    # Some distance functions are provided for every platform
    assert simd.to_int(simd.tanimoto_b1x8_naive) != 0
    assert simd.to_int(simd.tanimoto_maccs_naive) != 0

    # Arm Neon variants should be precompiled for any 64-bit Arm machine:
    if platform.machine() == "arm64":
        assert simd.to_int(simd.cos_f16x4_neon) != 0
        assert simd.to_int(simd.cos_i8x16_neon) != 0
        assert simd.to_int(simd.cos_f32x4_neon) != 0
        assert simd.to_int(simd.dot_f32x4_neon) != 0
        assert simd.to_int(simd.tanimoto_maccs_neon) != 0

    # Arm SVE variants should be precompiled for 64-bit Arm machines running Linux:
    if platform.machine() == "arm64" and platform.system().lower() == "linux":
        assert simd.to_int(simd.dot_f32_sve) != 0
        assert simd.to_int(simd.cos_f32_sve) != 0
        assert simd.to_int(simd.cos_f16_sve) != 0
        assert simd.to_int(simd.l2sq_f32_sve) != 0
        assert simd.to_int(simd.l2sq_f16_sve) != 0
        assert simd.to_int(simd.hamming_b1x8_sve) != 0
        assert simd.to_int(simd.hamming_b1x128_sve) != 0
        assert simd.to_int(simd.tanimoto_maccs_sve) != 0

    # x86 AVX2 and AVX-512 variants are precompiled for every 64-bit x86 platform:
    if platform.machine() == "x86_64":
        assert simd.to_int(simd.cos_f32x4_avx2) != 0
        assert simd.to_int(simd.cos_f16x16_avx512) != 0
        assert simd.to_int(simd.hamming_b1x128_avx512) != 0
        assert simd.to_int(simd.tanimoto_maccs_avx512) != 0
