{
    "variables": {"openssl_fips": ""},
    "targets": [
        {
            "target_name": "mathkong",
            "sources": ["javascript/lib.c", "c/lib.c"],
            "include_dirs": ["include"],
            "defines": [
                "MATHKONG_NATIVE_F16=0",
                "MATHKONG_NATIVE_BF16=0",
                "MATHKONG_DYNAMIC_DISPATCH=1",
            ],
            "cflags": [
                "-std=c11",
                "-ffast-math",
                "-Wno-unknown-pragmas",
                "-Wno-maybe-uninitialized",
                "-Wno-cast-function-type",
                "-Wno-switch",
            ],
            "conditions": [
                [
                    "OS=='mac'",
                    {
                        "xcode_settings": {
                            "MACOSX_DEPLOYMENT_TARGET": "11.0",
                            "OTHER_CFLAGS": ["-arch arm64", "-arch x86_64"],
                            "OTHER_LDFLAGS": ["-arch arm64", "-arch x86_64"],
                        }
                    },
                ]
            ],
        }
    ],
}
