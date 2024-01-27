fn main() {
    // Docs.rs only target is x86_64
    // `include/simsimd/types.h:119:9: error: _Float16 is not supported on this target`
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    cc::Build::new()
        .file("rust/lib.c")
        .include("include")
        .flag("-O3")
        .warnings(false)
        .compiler("/usr/bin/clang")
        .compile("simsimd");

    println!("cargo:rerun-if-changed=rust/lib.c");
    println!("cargo:rerun-if-changed=rust/lib.rs");
    println!("cargo:rerun-if-changed=include/simsimd/simsimd.h");
}
