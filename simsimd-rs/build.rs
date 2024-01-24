extern crate bindgen;

use bindgen::Builder;
use std::path::PathBuf;

static REPOSITORY: &str = "https://github.com/ashvardanian/SimSIMD.git";
static REPOSITORY_DIR: &str = "lib/simsimd/";

fn download_library() {
    let git_clone_status = std::process::Command::new("git")
        .arg("clone")
        .args(["--depth", "1"])
        .arg(REPOSITORY)
        .arg(REPOSITORY_DIR)
        .status()
        .expect("clone library repo");

    if !git_clone_status.success() {
        panic!("Could not clone SimSIMD repository.")
    }
}

fn main() {
    let repo_dir = std::path::Path::new(REPOSITORY_DIR);
    if !repo_dir.exists() {
        download_library()
    }

    let source = repo_dir.join("include/simsimd/simsimd.h");

    let bindings = Builder::default()
        .header("include/reexport.h")
        .allowlist_file("include/reexport.h")
        .trust_clang_mangling(true)
        .wrap_static_fns(true)
        .use_core()
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .unwrap();

    let output_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let obj_path = output_path.join("simsimd.o");

    let clang_output = std::process::Command::new("clang")
        .arg("-O3")
        .arg("-c")
        .arg("-o")
        .arg(&obj_path)
        .arg("src/simsimd.c")
        .arg("-include")
        .arg(source)
        .output()
        .unwrap();

    if !clang_output.status.success() {
        panic!(
            "Could not compile object file:\n{}",
            String::from_utf8_lossy(&clang_output.stderr)
        )
    }

    // Turn the object file into a static library
    #[cfg(not(target_os = "windows"))]
    let lib_output = std::process::Command::new("ar")
        .arg("rcs")
        .arg(output_path.join("libsimsimd.a"))
        .arg(obj_path)
        .output()
        .unwrap();
    #[cfg(target_os = "windows")]
    let lib_output = std::process::Command::new("LIB")
        .arg(obj_path)
        .arg(format!(
            "/OUT:{}",
            output_path.join("simsimd.lib").display()
        ))
        .output()
        .unwrap();

    if !lib_output.status.success() {
        panic!(
            "Could not emit library file:\n{}",
            String::from_utf8_lossy(&lib_output.stderr)
        );
    }

    println!("cargo:rustc-link-search={}", output_path.to_str().unwrap());
    println!("cargo:rustc-link-lib=static=simsimd");

    // Write the rust bindings.
    bindings
        .write_to_file(output_path.join("bindings.rs"))
        .expect("Cound not write bindings to the Rust file");
}
