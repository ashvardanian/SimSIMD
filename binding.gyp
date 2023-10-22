{
    "targets": [
        {
            "target_name": "simsimd",
            "sources": ["javascript/lib.c"],
            "include_dirs": ["include"],
            "defines": ["SIMSIMD_NATIVE_F16=0"],
            "cflags": [
                "-std=c11",
                "-ffast-math",
                "-Wno-unknown-pragmas",
                "-Wno-maybe-uninitialized",
                "-Wno-cast-function-type",
                "-Wno-switch",
            ],
        }
    ]
}
