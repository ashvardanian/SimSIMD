/* NumKong ISA probe: NEON BF16 (ARMv8.6-A bfloat16 dot product) */
#include <arm_neon.h>
int main(void) {
    bfloat16x8_t a = vdupq_n_bf16(1.0f);
    bfloat16x8_t b = vdupq_n_bf16(2.0f);
    float32x4_t c = vdupq_n_f32(0.0f);
    c = vbfdotq_f32(c, a, b);
    return vgetq_lane_f32(c, 0) > 0.0f ? 0 : 1;
}
