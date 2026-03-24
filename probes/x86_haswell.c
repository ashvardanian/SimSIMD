/* NumKong ISA probe: Haswell (AVX2 + FMA + F16C) */
#if !defined(__AVX2__)
#error "Feature not available"
#endif
#include <immintrin.h>
int main(void) {
    volatile int one = 1;
    __m256i a = _mm256_set1_epi32(one);
    __m256i b = _mm256_add_epi32(a, a);
    return _mm256_extract_epi32(b, 0) == 2 ? 0 : 1;
}
