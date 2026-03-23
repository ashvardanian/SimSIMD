# cmake/nk_loongarch_isa_probes.cmake — LoongArch ISA compiler-capability probes
#
# Detect whether the compiler can emit LASX (256-bit SIMD) instructions.
# On native LoongArch builds the probe compiles *and* runs to verify
# both compiler and hardware support.  Cross-compilation falls back to
# compile-only (runtime dispatch handles mismatches).
#
# Contract:
#   Input:  NK_BUILD_TEST, NK_BUILD_BENCH (from parent scope)
#   Output: nk_loongarch_isa_defs_ list (consumed by nk_test/nk_bench targets)

if (NOT (NK_BUILD_TEST OR NK_BUILD_BENCH))
    return()
endif ()

include(CheckSourceRuns)
include(CheckSourceCompiles)

set(nk_saved_required_flags_ "${CMAKE_REQUIRED_FLAGS}")
set(nk_saved_try_compile_config_ "${CMAKE_TRY_COMPILE_CONFIGURATION}")
set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")

macro (nk_loongarch_isa_probe_ var_ flags_ source_)
    set(CMAKE_REQUIRED_FLAGS "${flags_}")
    if (CMAKE_CROSSCOMPILING)
        check_source_compiles(C "${source_}" ${var_})
    else ()
        check_source_runs(C "${source_}" ${var_})
    endif ()
endmacro ()

# LASX — 256-bit Loongson Advanced SIMD Extension
nk_loongarch_isa_probe_(
    nk_target_loongsonasx_
    "-mlasx"
    "
    #include <lasxintrin.h>
    int main(void) {
        __m256i a = __lasx_xvreplgr2vr_w(1);
        __m256i b = __lasx_xvreplgr2vr_w(2);
        __m256i c = __lasx_xvadd_w(a, b);
        int r = __lasx_xvpickve2gr_w(c, 0);
        return r == 3 ? 0 : 1;
    }
"
)

set(CMAKE_REQUIRED_FLAGS "${nk_saved_required_flags_}")
set(CMAKE_TRY_COMPILE_CONFIGURATION "${nk_saved_try_compile_config_}")

set(nk_loongarch_isa_defs_ "")
foreach (isa_ IN ITEMS LOONGSONASX)
    string(TOLOWER "${isa_}" isa_lower_)
    if (nk_target_${isa_lower_}_)
        list(APPEND nk_loongarch_isa_defs_ "NK_TARGET_${isa_}=1")
    else ()
        list(APPEND nk_loongarch_isa_defs_ "NK_TARGET_${isa_}=0")
    endif ()
endforeach ()

if (nk_loongarch_isa_defs_)
    message(STATUS "LoongArch ISA compiler probes (for nk_test/nk_bench): ${nk_loongarch_isa_defs_}")
endif ()
