{
    "variables": {
        "openssl_fips": ""
    },
    "targets": [
        {
            "target_name": "numkong",
            "sources": [
                "javascript/numkong.c",
                "c/numkong.c",
                "c/dispatch_f64.c",
                "c/dispatch_f32.c",
                "c/dispatch_f16.c",
                "c/dispatch_bf16.c",
                "c/dispatch_i8.c",
                "c/dispatch_u8.c",
                "c/dispatch_u1.c",
                "c/dispatch_e4m3.c",
                "c/dispatch_e5m2.c",
                "c/dispatch_other.c",
                "c/dispatch_f64c.c",
                "c/dispatch_f32c.c",
                "c/dispatch_f16c.c",
                "c/dispatch_bf16c.c",
                "c/dispatch_i16.c",
                "c/dispatch_i32.c",
                "c/dispatch_i64.c",
                "c/dispatch_u16.c",
                "c/dispatch_u32.c",
                "c/dispatch_u64.c",
                "c/dispatch_i4.c",
                "c/dispatch_u4.c",
                "c/dispatch_e2m3.c",
                "c/dispatch_e3m2.c",
            ],
            "include_dirs": [
                "include"
            ],
            "defines": [
                "NK_NATIVE_F16=0",
                "NK_NATIVE_BF16=0",
                "NK_DYNAMIC_DISPATCH=1"
            ],
            "cflags": [
                "-std=c11",
                "-O3",
                "-Wno-unknown-pragmas",
                "-Wno-maybe-uninitialized",
                "-Wno-cast-function-type",
                "-Wno-switch",
                "-Wno-psabi",
                "-include",
                "build/nk_probes.h",
            ],
            "msvs_settings": {
                "VCCLCompilerTool": {
                    "ForcedIncludeFiles": [
                        "build/nk_probes.h"
                    ],
                },
            },
            "conditions": [
                [
                    "OS=='mac'",
                    {
                        "xcode_settings": {
                            "MACOSX_DEPLOYMENT_TARGET": "11.0"
                        }
                    }
                ],
            ],
        }
    ],
}