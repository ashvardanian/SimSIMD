/* NumKong ISA probe: NEON (AArch64 baseline SIMD) */
#if !defined(__ARM_NEON)
#error "NEON not available"
#endif
#include <arm_neon.h>
int main(void) {
    float32x4_t a = vdupq_n_f32(1.0f);
    float32x4_t b = vdupq_n_f32(2.0f);
    float32x4_t c = vaddq_f32(a, b);
    return vgetq_lane_f32(c, 0) > 0.0f ? 0 : 1;
}
