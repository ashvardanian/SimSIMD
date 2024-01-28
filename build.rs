fn main() {
    cc::Build::new()
        .file("rust/lib.c")
        .include("include")
        .flag("-O3")
        .flag("-DSIMSIMD_NATIVE_F16=0")
        .warnings(false)
        .compile("simsimd");

    println!("cargo:rerun-if-changed=rust/lib.c");
    println!("cargo:rerun-if-changed=rust/lib.rs");
    println!("cargo:rerun-if-changed=include/simsimd/simsimd.h");
}
