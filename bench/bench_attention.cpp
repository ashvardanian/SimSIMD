/**
 *  @brief Attention benchmarks - SME ISA.
 *  @file bench/bench_attention.cpp
 *  @author Ash Vardanian
 *  @date February 21, 2026
 */

#if NK_TARGET_SME
#include "numkong/attention/sme.h"
#endif

#include "bench.hpp"

#if NK_TARGET_SME

/**
 *  @brief Attention benchmark template: pack KV once, then benchmark the attention kernel.
 *
 *  Uses matrix_height as num_heads * query_len, matrix_width as kv_len, matrix_depth as head_dim.
 *  The benchmark reports scalar-ops as: num_heads * query_len * kv_len * head_dim * 2 (Q×K^T + P×V).
 *
 *  @tparam scalar_type_ The input/output element type (nk_bf16_t or nk_f16_t)
 */
template <typename scalar_type_>
void measure_attention_sme(                                                                                    //
    bm::State &state,                                                                                          //
    nk_size_t (*packed_kv_size_fn)(nk_size_t, nk_size_t, nk_size_t),                                           //
    void (*pack_kv_fn)(scalar_type_ const *, scalar_type_ const *, nk_size_t, nk_size_t, nk_size_t, nk_size_t, //
                       nk_size_t, void *),                                                                     //
    void (*attention_fn)(scalar_type_ const *, void const *, scalar_type_ *, nk_size_t, nk_size_t, nk_size_t,  //
                         nk_size_t, nk_size_t, nk_f32_t),                                                      //
    nk_size_t num_heads, nk_size_t query_len, nk_size_t kv_len, nk_size_t head_dim) {

    nk_size_t num_kv_heads = num_heads;
    nk_f32_t scale = 1.0f / std::sqrt((float)head_dim);

    // Allocate Q, K, V, output
    std::vector<scalar_type_> q(num_heads * query_len * head_dim);
    std::vector<scalar_type_> k(num_kv_heads * kv_len * head_dim);
    std::vector<scalar_type_> v(num_kv_heads * kv_len * head_dim);
    std::vector<scalar_type_> output(num_heads * query_len * head_dim);

    // Fill with random bit patterns (half-precision range)
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint16_t> dist(0x0000, 0x3C00); // [0, 1.0] in f16 range
    for (auto &x : q) {
        uint16_t bits = dist(gen);
        std::memcpy(&x, &bits, sizeof(x));
    }
    for (auto &x : k) {
        uint16_t bits = dist(gen);
        std::memcpy(&x, &bits, sizeof(x));
    }
    for (auto &x : v) {
        uint16_t bits = dist(gen);
        std::memcpy(&x, &bits, sizeof(x));
    }

    // Pack KV
    nk_size_t kv_packed_size = packed_kv_size_fn(num_kv_heads, head_dim, kv_len);
    std::vector<char> kv_packed(kv_packed_size, 0);
    nk_size_t k_stride = kv_len * head_dim;
    nk_size_t v_stride = kv_len * head_dim;
    pack_kv_fn(k.data(), v.data(), num_kv_heads, head_dim, kv_len, k_stride, v_stride, kv_packed.data());

    // Benchmark
    std::size_t iterations = 0;
    for (auto _ : state) {
        attention_fn(q.data(), kv_packed.data(), output.data(), num_heads, num_kv_heads, query_len, kv_len, head_dim,
                     scale);
        ++iterations;
        bm::DoNotOptimize(output.data());
    }

    // Q×K^T: num_heads * query_len * kv_len * head_dim * 2 (mul + add)
    // P×V:   num_heads * query_len * head_dim * kv_len * 2 (mul + add)
    // Total: 2 * 2 * num_heads * query_len * kv_len * head_dim
    state.counters["scalar-ops"] = bm::Counter(iterations * 4.0 * num_heads * query_len * kv_len * head_dim,
                                               bm::Counter::kIsRate);
}

template <typename scalar_type_>
void attention_sme_(                                                                                           //
    std::string name,                                                                                          //
    nk_size_t (*packed_kv_size_fn)(nk_size_t, nk_size_t, nk_size_t),                                           //
    void (*pack_kv_fn)(scalar_type_ const *, scalar_type_ const *, nk_size_t, nk_size_t, nk_size_t, nk_size_t, //
                       nk_size_t, void *),                                                                     //
    void (*attention_fn)(scalar_type_ const *, void const *, scalar_type_ *, nk_size_t, nk_size_t, nk_size_t,  //
                         nk_size_t, nk_size_t, nk_f32_t),                                                      //
    nk_size_t num_heads, nk_size_t query_len, nk_size_t kv_len, nk_size_t head_dim) {

    std::string bench_name = name + "<" + std::to_string(num_heads) + "h_" + std::to_string(query_len) + "q_" +
                             std::to_string(kv_len) + "kv_" + std::to_string(head_dim) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_attention_sme<scalar_type_>, packed_kv_size_fn, pack_kv_fn,
                          attention_fn, num_heads, query_len, kv_len, head_dim);
}

#endif // NK_TARGET_SME

void bench_attention() {
#if NK_TARGET_SME
    // Attention benchmarks: 8 heads, head_dim=128, varying query_len and kv_len
    // head_dim is always 128 (typical for LLaMA/Qwen); kv_len from matrix_height
    nk_size_t attn_num_heads = 8;
    nk_size_t attn_head_dim = 128;
    nk_size_t attn_query_len = 64;
    nk_size_t attn_kv_len = bench_config.matrix_height > 0 ? bench_config.matrix_height : 1024;
    attention_sme_<nk_bf16_t>("attention_bf16_sme", nk_attention_packed_kv_size_bf16_sme, nk_attention_pack_kv_bf16_sme,
                              nk_attention_bf16_sme, attn_num_heads, attn_query_len, attn_kv_len, attn_head_dim);
    attention_sme_<nk_f16_t>("attention_f16_sme", nk_attention_packed_kv_size_f16_sme, nk_attention_pack_kv_f16_sme,
                             nk_attention_f16_sme, attn_num_heads, attn_query_len, attn_kv_len, attn_head_dim);
#endif
}
