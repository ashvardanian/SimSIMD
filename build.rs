fn main() {
    cc::Build::new()
        .file("c/lib.c")
        .include("include")
        .flag("-O3")
        .flag("-DSIMSIMD_NATIVE_F16=0")
        .flag("-std=c99") // Enforce C99 standard
        .flag("-pedantic") // Ensure strict compliance with the C standard
        .warnings(false)
        .compile("simsimd");

    println!("cargo:rerun-if-changed=c/lib.c");
    println!("cargo:rerun-if-changed=rust/lib.rs");
    println!("cargo:rerun-if-changed=include/simsimd/simsimd.h");

    println!("cargo:rerun-if-changed=include/simsimd/dot.h");
    println!("cargo:rerun-if-changed=include/simsimd/spatial.h");
    println!("cargo:rerun-if-changed=include/simsimd/probability.h");
    println!("cargo:rerun-if-changed=include/simsimd/binary.h");
    println!("cargo:rerun-if-changed=include/simsimd/types.h");
}
