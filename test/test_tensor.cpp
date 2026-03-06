/**
 *  @brief C++ vector type instantiation tests.
 *  @file test/test_tensor.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */
#include <cassert>
#include <complex>

#include "test.hpp"

template <typename value_type_>
void test_vector_basics() {
    constexpr std::size_t dims_per_value = nk::dimensions_per_value<value_type_>();
    constexpr std::size_t test_dims = 64 * dims_per_value;
    auto v = make_vector<value_type_>(test_dims);
    assert(v.size() == test_dims);
    assert(v.size_values() == test_dims / dims_per_value);
    std::size_t count = 0;
    for (auto it = v.begin(); it != v.end(); ++it) ++count;
    assert(count == test_dims);
}

void test_signed_indexing() {
    auto v = make_vector<float>(100);
    v[50] = 3.14f;
    assert(v[50] == 3.14f && "float operator[] failed");
    v[-1] = 42.0f;
    assert(v[99] == 42.0f && "float signed indexing failed");
}

void test_move_semantics() {
    auto v1 = make_vector<nk::f32_t>(100);
    v1[50] = nk::f32_t(42.0f);

    nk::vector<nk::f32_t> v2 = std::move(v1);
    assert(v2.size() == 100 && "move ctor size mismatch");
    assert(v2[50] == nk::f32_t(42.0f) && "move ctor value mismatch");
    assert(v1.size() == 0 && "moved-from vector not empty");

    nk::vector<nk::f32_t> v3;
    v3 = std::move(v2);
    assert(v3.size() == 100 && "move assign size mismatch");
    assert(v3[50] == nk::f32_t(42.0f) && "move assign value mismatch");
}

void test_swap() {
    auto v1 = make_vector<nk::i8_t>(10);
    auto v2 = make_vector<nk::i8_t>(20);
    v1[0] = nk::i8_t(1);
    v2[0] = nk::i8_t(2);

    swap(v1, v2);
    assert(v1.size() == 20 && "swap v1 size mismatch");
    assert(v2.size() == 10 && "swap v2 size mismatch");
    assert(v1[0] == nk::i8_t(2) && "swap v1 value mismatch");
    assert(v2[0] == nk::i8_t(1) && "swap v2 value mismatch");
}

void test_view_span_rev() {
    auto v = make_vector<float>(5);
    v[0] = 1.0f;
    v[1] = 2.0f;
    v[2] = 3.0f;
    v[3] = 4.0f;
    v[4] = 5.0f;

    auto view = v.view();
    assert(view.size() == 5 && "view size mismatch");
    assert(view[-1] == 5.0f && "view signed indexing failed");

    auto span = v.span();
    span[0] = 10.0f;
    assert(v[0] == 10.0f && "span write-through failed");

    auto rev = view.rev();
    assert(rev[0] == 5.0f && "reversed view first element mismatch");
    assert(rev[4] == 10.0f && "reversed view last element mismatch");
}

void test_range_slicing() {
    auto v = make_vector<float>(5);
    v[0] = 1.0f;
    v[1] = 2.0f;
    v[2] = 3.0f;
    v[3] = 4.0f;
    v[4] = 5.0f;

    auto sub = v[nk::range(1, 4)];
    assert(sub.size() == 3 && "range slice size mismatch");
    assert(sub[0] == 2.0f && "range slice first element mismatch");
    assert(sub[2] == 4.0f && "range slice last element mismatch");
}

void test_sub_byte_i4x2() {
    auto v = make_vector<nk::i4x2_t>(100);
    assert(v.size() == 100 && "i4x2_t size mismatch");
    assert(v.size_values() == 50 && "i4x2_t size_values mismatch (should be dims/2)");

    v[0] = 5, v[1] = -3;
    assert(v[0] == 5 && "i4x2_t dim 0 mismatch");
    assert(v[1] == -3 && "i4x2_t dim 1 mismatch");
}

void test_sub_byte_u1x8() {
    auto v = make_vector<nk::u1x8_t>(64);
    assert(v.size() == 64 && "u1x8_t size mismatch");
    assert(v.size_values() == 8 && "u1x8_t size_values mismatch (should be dims/8)");

    v[0] = true, v[1] = false, v[7] = true;
    assert(v[0] == true && "u1x8_t dim 0 mismatch");
    assert(v[1] == false && "u1x8_t dim 1 mismatch");
    assert(v[7] == true && "u1x8_t dim 7 mismatch");
}

void test_custom_allocator() {
    using custom_alloc_t = nk::aligned_allocator<nk::f32_t, 128>;
    auto v = nk::vector<nk::f32_t, custom_alloc_t>::try_zeros(256);
    assert(v.size() == 256 && "custom allocator size mismatch");
    v[128] = nk::f32_t(99.0f);
    assert(v[128] == nk::f32_t(99.0f) && "custom allocator value mismatch");
}

void test_vector_types() {
    std::printf("Testing vector type instantiations...\n");

    // Template-based type coverage
    test_vector_basics<float>();
    test_vector_basics<double>();
    test_vector_basics<nk::f16_t>();
    test_vector_basics<nk::bf16_t>();
    test_vector_basics<nk::i8_t>();
    test_vector_basics<nk::f32c_t>();
    test_vector_basics<std::complex<double>>();
    test_vector_basics<nk::i4x2_t>();
    test_vector_basics<nk::u1x8_t>();
    std::printf("  vector basics (9 types):      OK\n");

    // Feature tests (non-template, using specific types)
    test_signed_indexing();
    std::printf("  signed indexing:              OK\n");

    test_move_semantics();
    std::printf("  move semantics:               OK\n");

    test_swap();
    std::printf("  swap:                         OK\n");

    test_view_span_rev();
    std::printf("  view/span/rev:                OK\n");

    test_range_slicing();
    std::printf("  range slicing:                OK\n");

    test_sub_byte_i4x2();
    test_sub_byte_u1x8();
    std::printf("  sub-byte i4x2/u1x8:           OK\n");

    test_custom_allocator();
    std::printf("  custom allocator:             OK\n");
}

/**
 *  @brief Explicit template instantiation test for all tensor-level operations.
 *
 *  Forces the compiler to fully instantiate every type × operation combination,
 *  catching signature mismatches, missing type traits, and implicit conversion errors
 *  that syntax-only checks miss.
 */
template <typename value_type_>
void test_tensor_ops_for_type() {
    using tensor_t = nk::tensor<value_type_>;

    // Create small test tensors
    auto a = tensor_t::try_zeros({4, 8});
    auto b = tensor_t::try_zeros({4, 8});
    assert(!a.empty() && !b.empty());

    auto av = a.view();
    auto bv = b.view();

    // Scalar reductions
    { [[maybe_unused]] auto r = nk::sum<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::moments<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::min<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::max<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::argmin<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::argmax<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::minmax<value_type_>(av); }

    // Axis reductions
    { [[maybe_unused]] auto r = nk::try_sum<value_type_>(av, 0); }
    { [[maybe_unused]] auto r = nk::try_sum<value_type_>(av, 1, nk::keep_dims_k); }
    { [[maybe_unused]] auto r = nk::try_moments<value_type_>(av, 1); }
    { [[maybe_unused]] auto r = nk::try_minmax<value_type_>(av, 0); }
    { [[maybe_unused]] auto r = nk::try_minmax<value_type_>(av, 1, nk::keep_dims_k); }
    { [[maybe_unused]] auto r = nk::try_min<value_type_>(av, 0); }
    { [[maybe_unused]] auto r = nk::try_min<value_type_>(av, 1, nk::keep_dims_k); }
    { [[maybe_unused]] auto r = nk::try_max<value_type_>(av, 0); }
    { [[maybe_unused]] auto r = nk::try_max<value_type_>(av, 1, nk::keep_dims_k); }

    // Elementwise binary
    { [[maybe_unused]] auto r = nk::try_add<value_type_>(av, bv); }
    { [[maybe_unused]] auto r = nk::try_sub<value_type_>(av, bv); }
    { [[maybe_unused]] auto r = nk::try_mul<value_type_>(av, bv); }

    // Elementwise binary with scalar
    using scale_t = typename value_type_::scale_t;
    scale_t scalar {1};
    { [[maybe_unused]] auto r = nk::try_add<value_type_>(av, scalar); }
    { [[maybe_unused]] auto r = nk::try_sub<value_type_>(av, scalar); }
    { [[maybe_unused]] auto r = nk::try_mul<value_type_>(av, scalar); }

    // Elementwise into
    auto out = tensor_t::try_zeros({4, 8});
    { [[maybe_unused]] bool ok = nk::add_into<value_type_>(av, bv, out.span()); }
    { [[maybe_unused]] bool ok = nk::sub_into<value_type_>(av, bv, out.span()); }
    { [[maybe_unused]] bool ok = nk::mul_into<value_type_>(av, bv, out.span()); }
    { [[maybe_unused]] bool ok = nk::add_into<value_type_>(av, scalar, out.span()); }
    { [[maybe_unused]] bool ok = nk::sub_into<value_type_>(av, scalar, out.span()); }
    { [[maybe_unused]] bool ok = nk::mul_into<value_type_>(av, scalar, out.span()); }

    // Affine
    scale_t alpha {1}, beta {0};
    { [[maybe_unused]] auto r = nk::try_scale<value_type_>(av, alpha, beta); }
    { [[maybe_unused]] auto r = nk::try_blend<value_type_>(av, bv, alpha, beta); }
    { [[maybe_unused]] auto r = nk::try_fma<value_type_>(av, bv, av, alpha, beta); }
    { [[maybe_unused]] bool ok = nk::scale_into<value_type_>(av, alpha, beta, out.span()); }
    { [[maybe_unused]] bool ok = nk::blend_into<value_type_>(av, bv, alpha, beta, out.span()); }
    { [[maybe_unused]] bool ok = nk::fma_into<value_type_>(av, bv, av, alpha, beta, out.span()); }
}

template <typename value_type_>
void test_tensor_trig_for_type() {
    using tensor_t = nk::tensor<value_type_>;
    auto a = tensor_t::try_zeros({4, 8});
    auto out = tensor_t::try_zeros({4, 8});
    auto av = a.view();

    { [[maybe_unused]] auto r = nk::try_sin<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::try_cos<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::try_atan<value_type_>(av); }
    { [[maybe_unused]] bool ok = nk::sin_into<value_type_>(av, out.span()); }
    { [[maybe_unused]] bool ok = nk::cos_into<value_type_>(av, out.span()); }
    { [[maybe_unused]] bool ok = nk::atan_into<value_type_>(av, out.span()); }
}

template <typename value_type_>
void test_tensor_symmetric_for_type() {
    using tensor_t = nk::tensor<value_type_>;
    auto a = tensor_t::try_zeros({4, 8});
    auto am = a.as_matrix_view();

    { [[maybe_unused]] auto r = nk::try_dots_symmetric<value_type_>(am); }
    { [[maybe_unused]] auto r = nk::try_angulars_symmetric<value_type_>(am); }
    { [[maybe_unused]] auto r = nk::try_euclideans_symmetric<value_type_>(am); }
}

template <typename value_type_>
void test_tensor_packed_for_type() {
    using tensor_t = nk::tensor<value_type_>;
    auto a = tensor_t::try_zeros({4, 8});
    auto b = tensor_t::try_zeros({6, 8});

    // packed_matrix
    auto bm = b.as_matrix_view();
    auto packed = nk::packed_matrix<value_type_, nk::aligned_allocator<char>>::try_pack(bm);
    auto am = a.as_matrix_view();
    auto result = nk::matrix<typename value_type_::dot_result_t>::try_zeros({4, 6});
    packed.multiply(am, result.span());
}

template <typename value_type_>
void test_tensor_maxsim_for_type() {
    using tensor_t = nk::tensor<value_type_>;
    auto q = tensor_t::try_zeros({3, 16});
    auto d = tensor_t::try_zeros({5, 16});

    auto qm = q.as_matrix_view();
    auto dm = d.as_matrix_view();

    auto pq = nk::packed_maxsim<value_type_>::try_pack(qm);
    auto pd = nk::packed_maxsim<value_type_>::try_pack(dm);
    { [[maybe_unused]] auto r = nk::maxsim(pq, pd); }
}

void test_tensor_ops() {
    std::printf("Testing tensor op instantiations...\n");

    // Core numeric types: all operations
    test_tensor_ops_for_type<nk::f32_t>();
    test_tensor_ops_for_type<nk::f64_t>();
    test_tensor_ops_for_type<nk::f16_t>();
    test_tensor_ops_for_type<nk::bf16_t>();
    test_tensor_ops_for_type<nk::i8_t>();
    test_tensor_ops_for_type<nk::u8_t>();
    std::printf("  ops (6 types):                OK\n");

    // Trig (float-capable types)
    test_tensor_trig_for_type<nk::f32_t>();
    test_tensor_trig_for_type<nk::f64_t>();
    test_tensor_trig_for_type<nk::f16_t>();
    test_tensor_trig_for_type<nk::bf16_t>();
    std::printf("  trig (4 types):               OK\n");

    // Symmetric distances
    test_tensor_symmetric_for_type<nk::f32_t>();
    test_tensor_symmetric_for_type<nk::f64_t>();
    test_tensor_symmetric_for_type<nk::f16_t>();
    test_tensor_symmetric_for_type<nk::bf16_t>();
    test_tensor_symmetric_for_type<nk::i8_t>();
    std::printf("  symmetric dist (5 types):     OK\n");

    // Packed GEMM
    test_tensor_packed_for_type<nk::f32_t>();
    test_tensor_packed_for_type<nk::f64_t>();
    test_tensor_packed_for_type<nk::f16_t>();
    test_tensor_packed_for_type<nk::bf16_t>();
    test_tensor_packed_for_type<nk::i8_t>();
    std::printf("  packed GEMM (5 types):        OK\n");

    // MaxSim (bf16, f32, f16 only)
    test_tensor_maxsim_for_type<nk::bf16_t>();
    test_tensor_maxsim_for_type<nk::f32_t>();
    test_tensor_maxsim_for_type<nk::f16_t>();
    std::printf("  packed MaxSim (3 types):      OK\n");
}
