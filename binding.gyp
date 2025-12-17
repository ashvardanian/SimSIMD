{
    "variables": {"openssl_fips": ""},
    "targets": [
        {
            "target_name": "simsimd",
            "sources": ["javascript/lib.c", "c/lib.c"],
            "include_dirs": ["include"],
            "defines": ["SIMSIMD_NATIVE_F16=0", "SIMSIMD_NATIVE_BF16=0", "SIMSIMD_DYNAMIC_DISPATCH=1"],
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
                    "OS=='linux' or OS=='freebsd'",
                    {
                        "conditions": [
                            [
                                "target_arch=='x64'",
                                {
                                    "defines": [
                                        "SIMSIMD_TARGET_HASWELL=1",
                                        "SIMSIMD_TARGET_SKYLAKE=1",
                                        "SIMSIMD_TARGET_ICE=1",
                                        "SIMSIMD_TARGET_GENOA=1",
                                        "SIMSIMD_TARGET_SAPPHIRE=1",
                                        "SIMSIMD_TARGET_TURIN=1",
                                        "SIMSIMD_TARGET_SIERRA=0",
                                        "SIMSIMD_TARGET_NEON=0",
                                        "SIMSIMD_TARGET_NEON_I8=0",
                                        "SIMSIMD_TARGET_NEON_F16=0",
                                        "SIMSIMD_TARGET_NEON_BF16=0",
                                        "SIMSIMD_TARGET_SVE=0",
                                        "SIMSIMD_TARGET_SVE_I8=0",
                                        "SIMSIMD_TARGET_SVE_F16=0",
                                        "SIMSIMD_TARGET_SVE_BF16=0",
                                        "SIMSIMD_TARGET_SVE2=0",
                                    ]
                                },
                            ],
                            [
                                "target_arch=='arm64'",
                                {
                                    "defines": [
                                        "SIMSIMD_TARGET_HASWELL=0",
                                        "SIMSIMD_TARGET_SKYLAKE=0",
                                        "SIMSIMD_TARGET_ICE=0",
                                        "SIMSIMD_TARGET_GENOA=0",
                                        "SIMSIMD_TARGET_SAPPHIRE=0",
                                        "SIMSIMD_TARGET_TURIN=0",
                                        "SIMSIMD_TARGET_SIERRA=0",
                                        "SIMSIMD_TARGET_NEON=1",
                                        "SIMSIMD_TARGET_NEON_I8=1",
                                        "SIMSIMD_TARGET_NEON_F16=1",
                                        "SIMSIMD_TARGET_NEON_BF16=1",
                                        "SIMSIMD_TARGET_SVE=1",
                                        "SIMSIMD_TARGET_SVE_I8=1",
                                        "SIMSIMD_TARGET_SVE_F16=1",
                                        "SIMSIMD_TARGET_SVE_BF16=1",
                                        "SIMSIMD_TARGET_SVE2=1",
                                    ]
                                },
                            ],
                        ]
                    },
                ],
                [
                    "OS=='mac'",
                    {
                        "xcode_settings": {"MACOSX_DEPLOYMENT_TARGET": "11.0"},
                        "conditions": [
                            [
                                "target_arch=='x64'",
                                {
                                    "defines": [
                                        "SIMSIMD_TARGET_HASWELL=1",
                                        "SIMSIMD_TARGET_SKYLAKE=0",
                                        "SIMSIMD_TARGET_ICE=0",
                                        "SIMSIMD_TARGET_GENOA=0",
                                        "SIMSIMD_TARGET_SAPPHIRE=0",
                                        "SIMSIMD_TARGET_TURIN=0",
                                        "SIMSIMD_TARGET_SIERRA=0",
                                        "SIMSIMD_TARGET_NEON=0",
                                        "SIMSIMD_TARGET_NEON_I8=0",
                                        "SIMSIMD_TARGET_NEON_F16=0",
                                        "SIMSIMD_TARGET_NEON_BF16=0",
                                        "SIMSIMD_TARGET_SVE=0",
                                        "SIMSIMD_TARGET_SVE_I8=0",
                                        "SIMSIMD_TARGET_SVE_F16=0",
                                        "SIMSIMD_TARGET_SVE_BF16=0",
                                        "SIMSIMD_TARGET_SVE2=0",
                                    ]
                                },
                            ],
                            [
                                "target_arch=='arm64'",
                                {
                                    "defines": [
                                        "SIMSIMD_TARGET_HASWELL=0",
                                        "SIMSIMD_TARGET_SKYLAKE=0",
                                        "SIMSIMD_TARGET_ICE=0",
                                        "SIMSIMD_TARGET_GENOA=0",
                                        "SIMSIMD_TARGET_SAPPHIRE=0",
                                        "SIMSIMD_TARGET_TURIN=0",
                                        "SIMSIMD_TARGET_SIERRA=0",
                                        "SIMSIMD_TARGET_NEON=1",
                                        "SIMSIMD_TARGET_NEON_I8=1",
                                        "SIMSIMD_TARGET_NEON_F16=1",
                                        "SIMSIMD_TARGET_NEON_BF16=1",
                                        "SIMSIMD_TARGET_SVE=0",
                                        "SIMSIMD_TARGET_SVE_I8=0",
                                        "SIMSIMD_TARGET_SVE_F16=0",
                                        "SIMSIMD_TARGET_SVE_BF16=0",
                                        "SIMSIMD_TARGET_SVE2=0",
                                    ]
                                },
                            ],
                        ],
                    },
                ],
                [
                    "OS=='win'",
                    {
                        "conditions": [
                            [
                                "target_arch=='x64'",
                                {
                                    "defines": [
                                        "SIMSIMD_TARGET_HASWELL=1",
                                        "SIMSIMD_TARGET_SKYLAKE=1",
                                        "SIMSIMD_TARGET_ICE=1",
                                        "SIMSIMD_TARGET_GENOA=0",
                                        "SIMSIMD_TARGET_SAPPHIRE=0",
                                        "SIMSIMD_TARGET_TURIN=0",
                                        "SIMSIMD_TARGET_SIERRA=0",
                                        "SIMSIMD_TARGET_NEON=0",
                                        "SIMSIMD_TARGET_NEON_I8=0",
                                        "SIMSIMD_TARGET_NEON_F16=0",
                                        "SIMSIMD_TARGET_NEON_BF16=0",
                                        "SIMSIMD_TARGET_SVE=0",
                                        "SIMSIMD_TARGET_SVE_I8=0",
                                        "SIMSIMD_TARGET_SVE_F16=0",
                                        "SIMSIMD_TARGET_SVE_BF16=0",
                                        "SIMSIMD_TARGET_SVE2=0",
                                    ]
                                },
                            ],
                            [
                                "target_arch=='arm64'",
                                {
                                    "defines": [
                                        "SIMSIMD_TARGET_HASWELL=0",
                                        "SIMSIMD_TARGET_SKYLAKE=0",
                                        "SIMSIMD_TARGET_ICE=0",
                                        "SIMSIMD_TARGET_GENOA=0",
                                        "SIMSIMD_TARGET_SAPPHIRE=0",
                                        "SIMSIMD_TARGET_TURIN=0",
                                        "SIMSIMD_TARGET_SIERRA=0",
                                        "SIMSIMD_TARGET_NEON=1",
                                        "SIMSIMD_TARGET_NEON_I8=1",
                                        "SIMSIMD_TARGET_NEON_F16=1",
                                        "SIMSIMD_TARGET_NEON_BF16=1",
                                        "SIMSIMD_TARGET_SVE=0",
                                        "SIMSIMD_TARGET_SVE_I8=0",
                                        "SIMSIMD_TARGET_SVE_F16=0",
                                        "SIMSIMD_TARGET_SVE_BF16=0",
                                        "SIMSIMD_TARGET_SVE2=0",
                                    ]
                                },
                            ],
                        ]
                    },
                ],
            ],
        }
    ],
}
