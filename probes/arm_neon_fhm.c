/* NumKong ISA probe: NEON FHM (ARMv8.2-A FP16 fused multiply-add) */
#if !defined(__ARM_FEATURE_FP16_FML) && !(defined(__ARM_ARCH) && __ARM_ARCH >= 804)
#error "NEON FHM not available"
#endif
#include <arm_neon.h>
int main(void) {
    float16x4_t a = vdup_n_f16(1.0f);
    float16x4_t b = vdup_n_f16(2.0f);
    float32x4_t c = vdupq_n_f32(0.0f);
    c = vfmlalq_low_f16(c, a, b);
    return vgetq_lane_f32(c, 0) > 0.0f ? 0 : 1;
}
