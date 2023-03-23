#include <benchmark/benchmark.h>

#include <simsimd/simsimd.hpp>

namespace bm = benchmark;
using namespace av::simsimd;

static constexpr std::uint16_t threads_k = 1;
static constexpr std::uint16_t time_k = 100;

template <typename metric_at, typename scalar_at, std::size_t dimensions_ak> //
static void measure(bm::State& state) {

    alignas(64) scalar_at a[dimensions_ak]{};
    alignas(64) scalar_at b[dimensions_ak]{};
    scalar_at c{};

    std::fill_n(a, dimensions_ak, 1);
    std::fill_n(b, dimensions_ak, 2);

    for (auto _ : state)
        bm::DoNotOptimize((c = metric_at{}(a, b, dimensions_ak)));

    state.SetBytesProcessed(state.iterations() * dimensions_ak * sizeof(scalar_at) * 2u);
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE(measure, cosine_similarity_t, f32_t, 16)->Threads(threads_k)->MinTime(time_k);
BENCHMARK_TEMPLATE(measure, cosine_similarity_t, f32_t, 256)->Threads(threads_k)->MinTime(time_k);
BENCHMARK_TEMPLATE(measure, cosine_similarity_f32x4k_t, f32_t, 16)->Threads(threads_k)->MinTime(time_k);
BENCHMARK_TEMPLATE(measure, cosine_similarity_f32x4k_t, f32_t, 256)->Threads(threads_k)->MinTime(time_k);

BENCHMARK_MAIN();
