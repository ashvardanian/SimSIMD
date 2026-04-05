# cmake/nk_arm32_isa_probes.cmake — ARM32 ISA compiler-capability probes
#
# Detect which NEON extensions the compiler can emit for 32-bit Arm.
# Probe source lives in probes/arm_*.c — shared with setup.py and build.rs.

set(nk_native_flags_ "-march=native")
include(cmake/nk_isa_probe.cmake)

nk_isa_probes_begin_()

# FEAT_AdvSIMD (baseline NEON on 32-bit Arm)
nk_isa_probe_(nk_target_neon "" "-mfpu=neon" "probes/arm_neon.c")
# FEAT_DotProd
nk_isa_probe_(nk_target_neonsdot "" "-march=armv8.2-a+dotprod -mfpu=auto" "probes/arm_neon_sdot.c")
# FEAT_FP16
nk_isa_probe_(nk_target_neonhalf "" "-march=armv8.2-a+fp16 -mfpu=auto" "probes/arm_neon_half.c")
# FEAT_FHM
nk_isa_probe_(nk_target_neonfhm "" "-march=armv8.2-a+fp16fml -mfpu=auto" "probes/arm_neon_fhm.c")

nk_isa_probes_end_()

nk_build_isa_defs_(
    arm32
    "Arm32"
    "NEON;NEONSDOT;NEONHALF;NEONFHM"
)
