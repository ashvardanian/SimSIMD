/* NumKong ISA probe: NEON SDOT (ARMv8.2-A dot product) */
#if !defined(__ARM_FEATURE_DOTPROD) && !(defined(__ARM_ARCH) && __ARM_ARCH >= 804)
#error "NEON SDOT not available"
#endif
#include <arm_neon.h>
int main(void) {
    int8x16_t a = vdupq_n_s8(1);
    int8x16_t b = vdupq_n_s8(2);
    int32x4_t c = vdupq_n_s32(0);
    c = vdotq_s32(c, a, b);
    return vgetq_lane_s32(c, 0) > 0 ? 0 : 1;
}
