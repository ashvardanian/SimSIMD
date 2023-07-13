import simsimd as sisi


def test_non_null():
    assert sisi.to_int(sisi.dot_f32_sve) != 0
    assert sisi.to_int(sisi.dot_f32x4_neon) != 0
    assert sisi.to_int(sisi.cos_f32_sve) != 0
    assert sisi.to_int(sisi.cos_f16_sve) != 0
    assert sisi.to_int(sisi.cos_f16x4_neon) != 0
    assert sisi.to_int(sisi.cos_i8x16_neon) != 0
    assert sisi.to_int(sisi.cos_f32x4_neon) != 0
    assert sisi.to_int(sisi.cos_f16x16_avx512) != 0
    assert sisi.to_int(sisi.cos_f32x4_avx2) != 0
    assert sisi.to_int(sisi.l2sq_f32_sve) != 0
    assert sisi.to_int(sisi.l2sq_f16_sve) != 0
    assert sisi.to_int(sisi.hamming_b1x8_sve) != 0
    assert sisi.to_int(sisi.hamming_b1x128_sve) != 0
    assert sisi.to_int(sisi.hamming_b1x128_avx512) != 0
    assert sisi.to_int(sisi.tanimoto_b1x8_naive) != 0
    assert sisi.to_int(sisi.tanimoto_maccs_naive) != 0
    assert sisi.to_int(sisi.tanimoto_maccs_neon) != 0
    assert sisi.to_int(sisi.tanimoto_maccs_sve) != 0
    assert sisi.to_int(sisi.tanimoto_maccs_avx512) != 0
