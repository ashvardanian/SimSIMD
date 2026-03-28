/* NumKong ISA probe: NEON F16 (ARMv8.2-A half-precision) */
#include <arm_neon.h>
int main(void) {
    float16x8_t a = vdupq_n_f16(1.0f);
    float16x8_t b = vdupq_n_f16(2.0f);
    float16x8_t c = vaddq_f16(a, b);
    return vgetq_lane_f16(c, 0) > 0.0f ? 0 : 1;
}
