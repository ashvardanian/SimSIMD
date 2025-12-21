use std::collections::HashMap;
use std::env;

fn main() {
    build_simsimd();
}

/// Build SimSIMD with dynamic SIMD dispatching.
/// Returns a HashMap of enabled compilation flags for potential reuse.
fn build_simsimd() -> HashMap<String, bool> {
    let mut flags = HashMap::<String, bool>::new();
    let mut build = cc::Build::new();

    build
        // Prefer portable flags to support MSVC and older toolchains
        .std("c99") // Enforce C99 standard when supported
        .file("c/lib.c")
        .include("include")
        .define("SIMSIMD_NATIVE_F16", "0")
        .define("SIMSIMD_NATIVE_BF16", "0")
        .define("SIMSIMD_DYNAMIC_DISPATCH", "1")
        .opt_level(3)
        .flag_if_supported("-pedantic") // Strict compliance when supported
        .warnings(false);

    // On 32-bit x86, ensure proper stack alignment for floating-point operations
    // See: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=38534
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch == "x86" {
        build.flag_if_supported("-mstackrealign");
        build.flag_if_supported("-mpreferred-stack-boundary=4");
    }

    // Set architecture-specific macros explicitly (like StringZilla)
    let target_bits = env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap_or_default();
    if target_arch == "x86_64" && target_bits == "64" {
        build.define("SIMSIMD_IS_64BIT_X86", "1");
        build.define("SIMSIMD_IS_64BIT_ARM", "0");
        flags.insert("SIMSIMD_IS_64BIT_X86".to_string(), true);
        flags.insert("SIMSIMD_IS_64BIT_ARM".to_string(), false);
    } else if target_arch == "aarch64" && target_bits == "64" {
        build.define("SIMSIMD_IS_64BIT_X86", "0");
        build.define("SIMSIMD_IS_64BIT_ARM", "1");
        flags.insert("SIMSIMD_IS_64BIT_X86".to_string(), false);
        flags.insert("SIMSIMD_IS_64BIT_ARM".to_string(), true);
    } else {
        build.define("SIMSIMD_IS_64BIT_X86", "0");
        build.define("SIMSIMD_IS_64BIT_ARM", "0");
        flags.insert("SIMSIMD_IS_64BIT_X86".to_string(), false);
        flags.insert("SIMSIMD_IS_64BIT_ARM".to_string(), false);
    }

    // Determine which backends to try based on target architecture.
    // The fallback mechanism will disable unsupported targets one by one.
    let flags_to_try = match target_arch.as_str() {
        "arm" | "aarch64" => vec![
            "SIMSIMD_TARGET_SVE2",
            "SIMSIMD_TARGET_SVE_BF16",
            "SIMSIMD_TARGET_SVE_F16",
            "SIMSIMD_TARGET_SVE_I8",
            "SIMSIMD_TARGET_SVE",
            "SIMSIMD_TARGET_NEON_BF16",
            "SIMSIMD_TARGET_NEON_F16",
            "SIMSIMD_TARGET_NEON_I8",
            "SIMSIMD_TARGET_NEON",
        ],
        "x86_64" => vec![
            "SIMSIMD_TARGET_SIERRA",
            "SIMSIMD_TARGET_TURIN",
            "SIMSIMD_TARGET_SAPPHIRE",
            "SIMSIMD_TARGET_GENOA",
            "SIMSIMD_TARGET_ICE",
            "SIMSIMD_TARGET_SKYLAKE",
            "SIMSIMD_TARGET_HASWELL",
        ],
        _ => vec![],
    };

    // Check environment variables to allow users to disable specific backends.
    // Usage: SIMSIMD_TARGET_NEON=0 SIMSIMD_TARGET_SVE=0 cargo build
    for flag in flags_to_try.iter() {
        let enabled = match env::var(flag) {
            Ok(val) => val != "0" && val.to_lowercase() != "false",
            Err(_) => true, // Default to enabled if not specified
        };

        if enabled {
            build.define(flag, "1");
            flags.insert(flag.to_string(), true);
        } else {
            build.define(flag, "0");
            flags.insert(flag.to_string(), false);
            println!("cargo:warning=Disabled {} via environment variable", flag);
        }
    }

    // Try compilation with all enabled backends
    if build.try_compile("simsimd").is_err() {
        println!("cargo:warning=Failed to compile SimSIMD with all SIMD backends...");

        // Fallback: disable backends one by one until compilation succeeds
        for flag in flags_to_try.iter() {
            build.define(flag, "0");
            flags.insert(flag.to_string(), false);

            if build.try_compile("simsimd").is_ok() {
                println!(
                    "cargo:warning=Successfully compiled after disabling {}",
                    flag
                );
                break;
            }

            println!(
                "cargo:warning=Failed to compile after disabling {}, trying next configuration...",
                flag
            );
        }
    }

    // Declare file dependencies
    println!("cargo:rerun-if-changed=c/lib.c");
    println!("cargo:rerun-if-changed=rust/lib.rs");
    println!("cargo:rerun-if-changed=include/simsimd/simsimd.h");
    println!("cargo:rerun-if-changed=include/simsimd/dot.h");
    println!("cargo:rerun-if-changed=include/simsimd/spatial.h");
    println!("cargo:rerun-if-changed=include/simsimd/probability.h");
    println!("cargo:rerun-if-changed=include/simsimd/binary.h");
    println!("cargo:rerun-if-changed=include/simsimd/types.h");

    // Rerun if environment variables change
    for flag in flags_to_try.iter() {
        println!("cargo:rerun-if-env-changed={}", flag);
    }

    flags
}
