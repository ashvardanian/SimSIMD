use std::collections::HashMap;
use std::env;

fn main() {
    build_simsimd();
}

/// Build NumKong with dynamic SIMD dispatching.
/// Returns a HashMap of enabled compilation flags for potential reuse.
fn build_simsimd() -> HashMap<String, bool> {
    let mut flags = HashMap::<String, bool>::new();
    let mut build = cc::Build::new();

    build
        // Prefer portable flags to support MSVC and older toolchains
        .std("c99") // Enforce C99 standard when supported
        .file("c/numkong.c")
        // Data type dispatch files
        .file("c/dispatch_f64.c")
        .file("c/dispatch_f32.c")
        .file("c/dispatch_f16.c")
        .file("c/dispatch_bf16.c")
        .file("c/dispatch_i8.c")
        .file("c/dispatch_u8.c")
        .file("c/dispatch_i4.c")
        .file("c/dispatch_u4.c")
        .file("c/dispatch_e4m3.c")
        .file("c/dispatch_e5m2.c")
        .file("c/dispatch_e2m3.c")
        .file("c/dispatch_e3m2.c")
        .file("c/dispatch_u1.c")
        // Complex type dispatch files
        .file("c/dispatch_f64c.c")
        .file("c/dispatch_f32c.c")
        .file("c/dispatch_f16c.c")
        .file("c/dispatch_bf16c.c")
        // Integer dispatch files
        .file("c/dispatch_i16.c")
        .file("c/dispatch_u16.c")
        .file("c/dispatch_i32.c")
        .file("c/dispatch_u32.c")
        .file("c/dispatch_i64.c")
        .file("c/dispatch_u64.c")
        // Special dispatch files
        .file("c/dispatch_cast.c")
        .include("include")
        .define("NK_NATIVE_F16", "0")
        .define("NK_NATIVE_BF16", "0")
        .define("NK_DYNAMIC_DISPATCH", "1")
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

    // Set architecture-specific macros explicitly
    let target_bits = env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap_or_default();
    if target_arch == "x86_64" && target_bits == "64" {
        build.define("NK_IS_64BIT_X86", "1");
        build.define("NK_IS_64BIT_ARM", "0");
        build.define("NK_IS_64BIT_RISCV", "0");
        flags.insert("NK_IS_64BIT_X86".to_string(), true);
        flags.insert("NK_IS_64BIT_ARM".to_string(), false);
        flags.insert("NK_IS_64BIT_RISCV".to_string(), false);
    } else if target_arch == "aarch64" && target_bits == "64" {
        build.define("NK_IS_64BIT_X86", "0");
        build.define("NK_IS_64BIT_ARM", "1");
        build.define("NK_IS_64BIT_RISCV", "0");
        flags.insert("NK_IS_64BIT_X86".to_string(), false);
        flags.insert("NK_IS_64BIT_ARM".to_string(), true);
        flags.insert("NK_IS_64BIT_RISCV".to_string(), false);
    } else if target_arch == "riscv64" && target_bits == "64" {
        build.define("NK_IS_64BIT_X86", "0");
        build.define("NK_IS_64BIT_ARM", "0");
        build.define("NK_IS_64BIT_RISCV", "1");
        flags.insert("NK_IS_64BIT_X86".to_string(), false);
        flags.insert("NK_IS_64BIT_ARM".to_string(), false);
        flags.insert("NK_IS_64BIT_RISCV".to_string(), true);
    } else {
        build.define("NK_IS_64BIT_X86", "0");
        build.define("NK_IS_64BIT_ARM", "0");
        build.define("NK_IS_64BIT_RISCV", "0");
        flags.insert("NK_IS_64BIT_X86".to_string(), false);
        flags.insert("NK_IS_64BIT_ARM".to_string(), false);
        flags.insert("NK_IS_64BIT_RISCV".to_string(), false);
    }

    // Determine which backends to try based on target architecture.
    // The fallback mechanism will disable unsupported targets one by one.
    let flags_to_try = match target_arch.as_str() {
        "arm" | "aarch64" => vec![
            // SME family
            "NK_TARGET_SMELUT2",
            "NK_TARGET_SMEBF16",
            "NK_TARGET_SMEBI32",
            "NK_TARGET_SMEHALF",
            "NK_TARGET_SMEFA64",
            "NK_TARGET_SMEF64",
            "NK_TARGET_SME2P1",
            "NK_TARGET_SME2",
            "NK_TARGET_SME",
            // SVE family
            "NK_TARGET_SVE2P1",
            "NK_TARGET_SVE2",
            "NK_TARGET_SVESDOT",
            "NK_TARGET_SVEBFDOT",
            "NK_TARGET_SVEHALF",
            "NK_TARGET_SVE",
            // NEON family
            "NK_TARGET_NEONFHM",
            "NK_TARGET_NEONBFDOT",
            "NK_TARGET_NEONSDOT",
            "NK_TARGET_NEONHALF",
            "NK_TARGET_NEON",
        ],
        "x86_64" => vec![
            // Most advanced first
            "NK_TARGET_GRANITEAMX",
            "NK_TARGET_SAPPHIREAMX",
            "NK_TARGET_SIERRA",
            "NK_TARGET_TURIN",
            "NK_TARGET_SAPPHIRE",
            "NK_TARGET_GENOA",
            "NK_TARGET_ICELAKE",
            "NK_TARGET_SKYLAKE",
            "NK_TARGET_HASWELL",
        ],
        "riscv64" => vec![
            // Most advanced first
            "NK_TARGET_RVVBB",
            "NK_TARGET_RVVBF16",
            "NK_TARGET_RVVHALF",
            "NK_TARGET_RVV",
        ],
        "wasm32" | "wasm64" => vec![
            //
            "NK_TARGET_V128RELAXED",
        ],
        _ => vec![],
    };

    // Check environment variables to allow users to disable specific backends.
    // Usage: NK_TARGET_NEON=0 NK_TARGET_SVE=0 cargo build
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
    if build.try_compile("numkong").is_err() {
        println!("cargo:warning=Failed to compile NumKong with all SIMD backends...");

        // Fallback: disable backends one by one until compilation succeeds
        for flag in flags_to_try.iter() {
            build.define(flag, "0");
            flags.insert(flag.to_string(), false);

            if build.try_compile("numkong").is_ok() {
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
    println!("cargo:rerun-if-changed=c/numkong.c");
    println!("cargo:rerun-if-changed=c/dispatch_f64.c");
    println!("cargo:rerun-if-changed=c/dispatch_f32.c");
    println!("cargo:rerun-if-changed=c/dispatch_f16.c");
    println!("cargo:rerun-if-changed=c/dispatch_bf16.c");
    println!("cargo:rerun-if-changed=c/dispatch_i8.c");
    println!("cargo:rerun-if-changed=c/dispatch_u8.c");
    println!("cargo:rerun-if-changed=c/dispatch_i4.c");
    println!("cargo:rerun-if-changed=c/dispatch_u4.c");
    println!("cargo:rerun-if-changed=c/dispatch_e4m3.c");
    println!("cargo:rerun-if-changed=c/dispatch_e5m2.c");
    println!("cargo:rerun-if-changed=c/dispatch_e2m3.c");
    println!("cargo:rerun-if-changed=c/dispatch_e3m2.c");
    println!("cargo:rerun-if-changed=c/dispatch_u1.c");
    println!("cargo:rerun-if-changed=c/dispatch_f64c.c");
    println!("cargo:rerun-if-changed=c/dispatch_f32c.c");
    println!("cargo:rerun-if-changed=c/dispatch_f16c.c");
    println!("cargo:rerun-if-changed=c/dispatch_bf16c.c");
    println!("cargo:rerun-if-changed=c/dispatch_i16.c");
    println!("cargo:rerun-if-changed=c/dispatch_u16.c");
    println!("cargo:rerun-if-changed=c/dispatch_i32.c");
    println!("cargo:rerun-if-changed=c/dispatch_u32.c");
    println!("cargo:rerun-if-changed=c/dispatch_i64.c");
    println!("cargo:rerun-if-changed=c/dispatch_u64.c");
    println!("cargo:rerun-if-changed=c/dispatch_cast.c");
    println!("cargo:rerun-if-changed=c/dispatch.h");
    println!("cargo:rerun-if-changed=rust/numkong.rs");
    // Top-level headers
    println!("cargo:rerun-if-changed=include/numkong/numkong.h");
    println!("cargo:rerun-if-changed=include/numkong/types.h");
    println!("cargo:rerun-if-changed=include/numkong/binary.h");
    println!("cargo:rerun-if-changed=include/numkong/curved.h");
    println!("cargo:rerun-if-changed=include/numkong/dot.h");
    println!("cargo:rerun-if-changed=include/numkong/dots.h");
    println!("cargo:rerun-if-changed=include/numkong/elementwise.h");
    println!("cargo:rerun-if-changed=include/numkong/geospatial.h");
    println!("cargo:rerun-if-changed=include/numkong/mesh.h");
    println!("cargo:rerun-if-changed=include/numkong/probability.h");
    println!("cargo:rerun-if-changed=include/numkong/reduce.h");
    println!("cargo:rerun-if-changed=include/numkong/sparse.h");
    println!("cargo:rerun-if-changed=include/numkong/spatial.h");
    println!("cargo:rerun-if-changed=include/numkong/trigonometry.h");
    // binary/
    println!("cargo:rerun-if-changed=include/numkong/binary/serial.h");
    println!("cargo:rerun-if-changed=include/numkong/binary/haswell.h");
    println!("cargo:rerun-if-changed=include/numkong/binary/ice.h");
    println!("cargo:rerun-if-changed=include/numkong/binary/neon.h");
    println!("cargo:rerun-if-changed=include/numkong/binary/sve.h");
    // dot/
    println!("cargo:rerun-if-changed=include/numkong/dot/serial.h");
    println!("cargo:rerun-if-changed=include/numkong/dot/haswell.h");
    println!("cargo:rerun-if-changed=include/numkong/dot/skylake.h");
    println!("cargo:rerun-if-changed=include/numkong/dot/ice.h");
    println!("cargo:rerun-if-changed=include/numkong/dot/genoa.h");
    println!("cargo:rerun-if-changed=include/numkong/dot/sapphire.h");
    println!("cargo:rerun-if-changed=include/numkong/dot/sierra.h");
    println!("cargo:rerun-if-changed=include/numkong/dot/neon.h");
    println!("cargo:rerun-if-changed=include/numkong/dot/neonhalf.h");
    println!("cargo:rerun-if-changed=include/numkong/dot/neonbfdot.h");
    println!("cargo:rerun-if-changed=include/numkong/dot/neonsdot.h");
    println!("cargo:rerun-if-changed=include/numkong/dot/neonfhm.h");
    println!("cargo:rerun-if-changed=include/numkong/dot/sve.h");
    println!("cargo:rerun-if-changed=include/numkong/dot/svehalf.h");
    println!("cargo:rerun-if-changed=include/numkong/dot/spacemit.h");
    println!("cargo:rerun-if-changed=include/numkong/dot/sifive.h");
    println!("cargo:rerun-if-changed=include/numkong/dot/xuantie.h");
    // dots/
    println!("cargo:rerun-if-changed=include/numkong/dots/haswell.h");
    println!("cargo:rerun-if-changed=include/numkong/dots/skylake.h");
    println!("cargo:rerun-if-changed=include/numkong/dots/ice.h");
    println!("cargo:rerun-if-changed=include/numkong/dots/genoa.h");
    println!("cargo:rerun-if-changed=include/numkong/dots/sapphire.h");
    println!("cargo:rerun-if-changed=include/numkong/dots/sapphire_amx.h");
    println!("cargo:rerun-if-changed=include/numkong/dots/sierra.h");
    println!("cargo:rerun-if-changed=include/numkong/dots/sve.h");
    println!("cargo:rerun-if-changed=include/numkong/dots/svehalf.h");
    // elementwise/
    println!("cargo:rerun-if-changed=include/numkong/elementwise/serial.h");
    println!("cargo:rerun-if-changed=include/numkong/elementwise/haswell.h");
    println!("cargo:rerun-if-changed=include/numkong/elementwise/skylake.h");
    println!("cargo:rerun-if-changed=include/numkong/elementwise/ice.h");
    println!("cargo:rerun-if-changed=include/numkong/elementwise/genoa.h");
    println!("cargo:rerun-if-changed=include/numkong/elementwise/sapphire.h");
    println!("cargo:rerun-if-changed=include/numkong/elementwise/sierra.h");
    println!("cargo:rerun-if-changed=include/numkong/elementwise/neonhalf.h");
    println!("cargo:rerun-if-changed=include/numkong/elementwise/neonsdot.h");
    println!("cargo:rerun-if-changed=include/numkong/elementwise/sve.h");
    println!("cargo:rerun-if-changed=include/numkong/elementwise/svehalf.h");
    // mesh/
    println!("cargo:rerun-if-changed=include/numkong/mesh/serial.h");
    println!("cargo:rerun-if-changed=include/numkong/mesh/haswell.h");
    println!("cargo:rerun-if-changed=include/numkong/mesh/skylake.h");
    println!("cargo:rerun-if-changed=include/numkong/mesh/ice.h");
    println!("cargo:rerun-if-changed=include/numkong/mesh/genoa.h");
    println!("cargo:rerun-if-changed=include/numkong/mesh/sapphire.h");
    println!("cargo:rerun-if-changed=include/numkong/mesh/sierra.h");
    println!("cargo:rerun-if-changed=include/numkong/mesh/neon.h");
    println!("cargo:rerun-if-changed=include/numkong/mesh/neonhalf.h");
    println!("cargo:rerun-if-changed=include/numkong/mesh/neonbfdot.h");
    println!("cargo:rerun-if-changed=include/numkong/mesh/neonsdot.h");
    // reduce/
    println!("cargo:rerun-if-changed=include/numkong/reduce/serial.h");
    println!("cargo:rerun-if-changed=include/numkong/reduce/haswell.h");
    println!("cargo:rerun-if-changed=include/numkong/reduce/skylake.h");
    println!("cargo:rerun-if-changed=include/numkong/reduce/ice.h");
    println!("cargo:rerun-if-changed=include/numkong/reduce/genoa.h");
    println!("cargo:rerun-if-changed=include/numkong/reduce/sapphire.h");
    println!("cargo:rerun-if-changed=include/numkong/reduce/sierra.h");
    println!("cargo:rerun-if-changed=include/numkong/reduce/neonhalf.h");
    println!("cargo:rerun-if-changed=include/numkong/reduce/neonbfdot.h");
    println!("cargo:rerun-if-changed=include/numkong/reduce/neonsdot.h");
    // spatial/
    println!("cargo:rerun-if-changed=include/numkong/spatial/serial.h");
    println!("cargo:rerun-if-changed=include/numkong/spatial/haswell.h");
    println!("cargo:rerun-if-changed=include/numkong/spatial/skylake.h");
    println!("cargo:rerun-if-changed=include/numkong/spatial/ice.h");
    println!("cargo:rerun-if-changed=include/numkong/spatial/genoa.h");
    println!("cargo:rerun-if-changed=include/numkong/spatial/sapphire.h");
    println!("cargo:rerun-if-changed=include/numkong/spatial/sierra.h");
    println!("cargo:rerun-if-changed=include/numkong/spatial/neon.h");
    println!("cargo:rerun-if-changed=include/numkong/spatial/neonhalf.h");
    println!("cargo:rerun-if-changed=include/numkong/spatial/neonbfdot.h");
    println!("cargo:rerun-if-changed=include/numkong/spatial/neonsdot.h");
    println!("cargo:rerun-if-changed=include/numkong/spatial/sve.h");
    println!("cargo:rerun-if-changed=include/numkong/spatial/svehalf.h");
    println!("cargo:rerun-if-changed=include/numkong/spatial/svebfdot.h");
    println!("cargo:rerun-if-changed=include/numkong/spatial/spacemit.h");
    println!("cargo:rerun-if-changed=include/numkong/spatial/sifive.h");
    println!("cargo:rerun-if-changed=include/numkong/spatial/xuantie.h");
    // trigonometry/
    println!("cargo:rerun-if-changed=include/numkong/trigonometry/serial.h");
    println!("cargo:rerun-if-changed=include/numkong/trigonometry/haswell.h");
    println!("cargo:rerun-if-changed=include/numkong/trigonometry/skylake.h");
    println!("cargo:rerun-if-changed=include/numkong/trigonometry/ice.h");
    println!("cargo:rerun-if-changed=include/numkong/trigonometry/genoa.h");
    println!("cargo:rerun-if-changed=include/numkong/trigonometry/sapphire.h");
    println!("cargo:rerun-if-changed=include/numkong/trigonometry/sierra.h");
    println!("cargo:rerun-if-changed=include/numkong/trigonometry/neon.h");
    println!("cargo:rerun-if-changed=include/numkong/trigonometry/neonhalf.h");
    println!("cargo:rerun-if-changed=include/numkong/trigonometry/neonbfdot.h");
    println!("cargo:rerun-if-changed=include/numkong/trigonometry/neonsdot.h");

    // Rerun if environment variables change
    for flag in flags_to_try.iter() {
        println!("cargo:rerun-if-env-changed={}", flag);
    }

    flags
}
