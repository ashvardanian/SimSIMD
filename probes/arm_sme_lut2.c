/* NumKong ISA probe: SME LUT2 (FEAT_SME_LUTv2) */
#if defined(_WIN32)
#error "SVE/SME not supported on Windows ARM"
#endif

#if !defined(__ARM_FEATURE_SME2)
#error "Feature not available"
#endif
#include <arm_sme.h>
__arm_new("zt0") __arm_locally_streaming int test_smelut2(void) {
    svuint8_t idx = svdup_u8(0);
    svuint8_t r = svluti2_lane_zt_u8(0, idx, 0);
    return (int)svaddv_u8(svptrue_b8(), r) == 0 ? 0 : 1;
}
int main(void) { return test_smelut2(); }
