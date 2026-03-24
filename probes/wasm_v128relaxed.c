/* NumKong ISA probe: WASM Relaxed SIMD (v128) */
#if !defined(__wasm_relaxed_simd__)
#error "WASM Relaxed SIMD not available"
#endif
#include <wasm_simd128.h>
int main(void) {
    v128_t a = wasm_f32x4_splat(1.0f);
    v128_t b = wasm_f32x4_splat(2.0f);
    v128_t c = wasm_f32x4_relaxed_madd(a, b, a);
    return wasm_f32x4_extract_lane(c, 0) > 0.0f ? 0 : 1;
}
