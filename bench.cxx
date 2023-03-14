#include <benchmark/benchmark.h>

#include <simsimd/simsimd.hpp>

namespace bm = benchmark;
using namespace av::simsimd;

template <typename metric_at, typename scalar_at, std::size_t dimensions_ak> //
static void measure(bm::State &state) {
    
    scalar_at a[dimensions_ak]{};
    scalar_at b[dimensions_ak]{};
    scalar_at c{};

    for (auto _ : state)
        c = metric_at{} (a, b, dimensions_ak);
    
}

BENCHMARK_TEMPLATE(measure, dot_product_t, f32_t, 16);
BENCHMARK_TEMPLATE(measure, dot_product_t, f32_t, 256);
BENCHMARK_TEMPLATE(measure, dot_product_f32x4k_t, f32_t, 16);
BENCHMARK_TEMPLATE(measure, dot_product_f32x4k_t, f32_t, 256);

BENCHMARK_MAIN();
