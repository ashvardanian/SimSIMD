CFLAGS_RELEASE = -O3 -DNDEBUG -ffast-math

CFLAGS_ARM_NEON = -march=armv8-a+simd+dotprod+fp16
CFLAGS_ARM_SVE = -march=armv8-a+sve+fp16
CFLAGS_X86_AVX2 = -mavx2 -mf16c -mfma
CFLAGS_X86_AVX512 = -mavx512fp16 -mavx512bf16 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512cd -mavx512vnni

all: arm_neon arm_sve x86_avx2 x86_avx512

# ARM Neon library
arm_neon: arm_neon_f16.o arm_neon_f32.o arm_neon_i8.o
	ar rcs libsimsimd_arm_neon.a $^

arm_neon_f16.o:
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_ARM_NEON) -c arm_neon_f16.c

arm_neon_f32.o:
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_ARM_NEON) -c arm_neon_f32.c

arm_neon_i8.o:
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_ARM_NEON) -c arm_neon_i8.c

# ARM SVE library
arm_sve: arm_sve_f16.o arm_sve_f32.o
	ar rcs libsimsimd_arm_sve.a $^

arm_sve_f16.o:
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_ARM_SVE) -c arm_sve_f16.c

arm_sve_f32.o:
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_ARM_SVE) -c arm_sve_f32.c

# x86 AVX2 library
x86_avx2: x86_avx2_f16.o x86_avx2_i8.o
	ar rcs libsimsimd_x86_avx2.a $^

x86_avx2_f16.o:
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_X86_AVX2) -c x86_avx2_f16.c

x86_avx2_i8.o:
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_X86_AVX2) -c x86_avx2_i8.c

# x86 AVX512 library
x86_avx512: x86_avx512_f16.o x86_avx512_i8.o
	ar rcs libsimsimd_x86_avx512.a $^

x86_avx512_f16.o:
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_X86_AVX512) -c x86_avx512_f16.c

x86_avx512_i8.o:
	$(CC) $(CFLAGS_RELEASE) $(CFLAGS_X86_AVX512) -c x86_avx512_i8.c

clean:
	rm -f *.o
	rm -f *.a
