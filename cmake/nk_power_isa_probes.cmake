# cmake/nk_power_isa_probes.cmake — Power ISA compiler-capability probes
#
# Detect which ISA extensions the host CPU can actually execute.  Results are
# used to override NK_TARGET_* on the nk_test and nk_bench targets so that
# ISA-specific kernels are only compiled when they can run on the build host.
# The nk_shared library keeps all ISAs enabled (its dispatch is runtime-selected
# via nk_capabilities() function pointers).
#
# Native builds: probes use check_source_runs to compile with per-ISA flags
# *and* verify the host CPU can execute the resulting binary.
# Cross-compilation: probes use check_source_compiles (can't run host binaries),
# relying on runtime dispatch in the library.
#
# Contract:
#   Input:  NK_BUILD_TEST, NK_BUILD_BENCH (from parent scope)
#   Output: nk_power_isa_defs_ list (consumed by nk_test/nk_bench targets)

if (NOT (NK_BUILD_TEST OR NK_BUILD_BENCH))
    return()
endif ()

include(CheckSourceRuns)
include(CheckSourceCompiles)

set(nk_saved_required_flags_ "${CMAKE_REQUIRED_FLAGS}")
set(nk_saved_try_compile_config_ "${CMAKE_TRY_COMPILE_CONFIGURATION}")
set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")

macro (nk_power_isa_probe_ var_ gcc_flags_ source_)
    if (CMAKE_CROSSCOMPILING)
        set(CMAKE_REQUIRED_FLAGS "${gcc_flags_}")
        check_source_compiles(C "${source_}" ${var_})
    else ()
        set(CMAKE_REQUIRED_FLAGS "${gcc_flags_}")
        check_source_runs(C "${source_}" ${var_})
    endif ()
endmacro ()

# Power VSX — 128-bit SIMD (POWER9+ baseline)
# Requires POWER9 for vec_extract, vec_xl_len, vec_cmpne, vec_extract_fp32_from_shorth.
nk_power_isa_probe_(
    nk_target_powervsx_ "-mcpu=power9" "
    #include <altivec.h>
    int main(void) {
        __vector float a = vec_splats(1.0f);
        __vector float b = vec_splats(2.0f);
        __vector float c = vec_madd(a, b, a);
        // vec_extract requires POWER9+
        return vec_extract(c, 0) == 3.0f ? 0 : 1;
    }
"
)

set(CMAKE_REQUIRED_FLAGS "${nk_saved_required_flags_}")
set(CMAKE_TRY_COMPILE_CONFIGURATION "${nk_saved_try_compile_config_}")

# Build the override list from probe results
set(nk_power_isa_defs_ "")
foreach (isa_ IN ITEMS POWERVSX)
    string(TOLOWER "${isa_}" isa_lower_)
    if (nk_target_${isa_lower_}_)
        list(APPEND nk_power_isa_defs_ "NK_TARGET_${isa_}=1")
    else ()
        list(APPEND nk_power_isa_defs_ "NK_TARGET_${isa_}=0")
    endif ()
endforeach ()

if (nk_power_isa_defs_)
    message(STATUS "Power ISA compiler probes (for nk_test/nk_bench): ${nk_power_isa_defs_}")
endif ()
