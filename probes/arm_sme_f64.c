/* NumKong ISA probe: SME F64 (FEAT_SME_F64F64) */
#if defined(_WIN32)
#error "SVE/SME not supported on Windows ARM"
#endif

#if !defined(__ARM_FEATURE_SME)
#error "Feature not available"
#endif
#include <arm_sme.h>
__arm_new("za") __arm_locally_streaming int test_smef64(void) {
    svfloat64_t a = svdup_f64(1.0);
    svbool_t p = svptrue_b64();
    svmopa_za64_f64_m(0, p, p, a, a);
    return 0;
}
int main(void) { return test_smef64(); }
