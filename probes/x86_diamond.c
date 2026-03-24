/* NumKong ISA probe: Diamond Rapids (AVX10.2) */
#if defined(__APPLE__)
#error "AVX-512 not available on macOS"
#endif

#if !defined(__AVX512FP16__)
#error "Feature not available"
#endif
#include <immintrin.h>
int main(void) {
    volatile int one = 1;
    __m256i acc = _mm256_setzero_si256();
    __m256i a = _mm256_set1_epi8((char)one);
    __m256i b = _mm256_set1_epi8((char)one);
    acc = _mm256_dpbe4ss_epi32(acc, a, b);
    return _mm256_extract_epi32(acc, 0) != 0 ? 0 : 1;
}
