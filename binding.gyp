{
    "targets": [
        {
            "target_name": "simsimd",
            "sources": ["javascript/lib.c"],
            "include_dirs": ["include"],
            "cflags": [
                "-std=c11",
                "-ffast-math",
                "-Wno-unknown-pragmas",
                "-Wno-maybe-uninitialized",
                "-Wno-cast-function-type",
                "-Wno-switch",
                "-DSIMSIMD_NATIVE_F16=0",
            ],
        }
    ]
}
