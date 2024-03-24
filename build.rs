fn main() {
    let mut build = cc::Build::new();

    build
        .file("c/lib.c")
        .include("include")
        .define("SIMSIMD_NATIVE_F16", "0")
        .flag("-O3")
        .flag("-std=c99") // Enforce C99 standard
        .flag("-pedantic") // Ensure strict compliance with the C standard
        .warnings(false);

    // Conditional compilation depending on the target operating system.
    if cfg!(target_os = "linux") {
        build
            .define("SIMSIMD_TARGET_NEON", "1")
            .define("SIMSIMD_TARGET_SVE", "1")
            .define("SIMSIMD_TARGET_HASWELL", "1")
            .define("SIMSIMD_TARGET_SKYLAKE", "1")
            .define("SIMSIMD_TARGET_ICE", "1")
            .define("SIMSIMD_TARGET_SAPPHIRE", "1");
    } else if cfg!(target_os = "macos") {
        build
            .define("SIMSIMD_TARGET_NEON", "1")
            .define("SIMSIMD_TARGET_SVE", "0")
            .define("SIMSIMD_TARGET_HASWELL", "1")
            .define("SIMSIMD_TARGET_SKYLAKE", "0")
            .define("SIMSIMD_TARGET_ICE", "0")
            .define("SIMSIMD_TARGET_SAPPHIRE", "0");
    } else if cfg!(target_os = "windows") {
        build
            .define("SIMSIMD_TARGET_NEON", "1")
            .define("SIMSIMD_TARGET_SVE", "0")
            .define("SIMSIMD_TARGET_HASWELL", "1")
            .define("SIMSIMD_TARGET_SKYLAKE", "1")
            .define("SIMSIMD_TARGET_ICE", "1")
            .define("SIMSIMD_TARGET_SAPPHIRE", "0");
    }

    build.compile("simsimd");

    println!("cargo:rerun-if-changed=c/lib.c");
    println!("cargo:rerun-if-changed=rust/lib.rs");
    println!("cargo:rerun-if-changed=include/simsimd/simsimd.h");

    println!("cargo:rerun-if-changed=include/simsimd/dot.h");
    println!("cargo:rerun-if-changed=include/simsimd/spatial.h");
    println!("cargo:rerun-if-changed=include/simsimd/probability.h");
    println!("cargo:rerun-if-changed=include/simsimd/binary.h");
    println!("cargo:rerun-if-changed=include/simsimd/types.h");
}
