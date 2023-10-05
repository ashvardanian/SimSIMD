
CFLAGS_RELEASE = -O3 -DNDEBUG -ffast-math

CFLAGS_ARM_NEON = -march=armv8-a+simd+dotprod+fp16
CFLAGS_ARM_SVE = -march=armv8-a+sve+fp16
CFLAGS_X86_AVX2 = -mavx2 -mf16c -mfma
CFLAGS_X86_AVX512 = -mavx512fp16 -mavx512bf16 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512cd -mavx512vnni

all: arm_neon arm_sve x86_avx2 x86_avx512

arm_neon:
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_ARM_NEON) -S arm_neon_f16.c -o arm_neon_f16.s
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_ARM_NEON) -S arm_neon_f32.c -o arm_neon_f32.s
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_ARM_NEON) -S arm_neon_i8.c -o arm_neon_i8.s

arm_sve:
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_ARM_SVE) -S arm_sve_f16.c -o arm_sve_f16.s
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_ARM_SVE) -S arm_sve_f32.c -o arm_sve_f32.s

x86_avx2:
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_X86_AVX2) -S x86_avx2_f16.c -o x86_avx2_f16.s
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_X86_AVX2) -S x86_avx2_i8.c -o x86_avx2_i8.s

x86_avx512:
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_X86_AVX512) -S x86_avx512_f16.c -o x86_avx512_f16.s
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_X86_AVX512) -S x86_avx512_i8.c -o x86_avx512_i8.s

clean:
	rm -f *.s
