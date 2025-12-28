{
    "variables": {"openssl_fips": ""},
    "targets": [
        {
            "target_name": "numkong",
            "sources": ["javascript/numkong.c", "c/numkong.c"],
            "include_dirs": ["include"],
            "defines": ["NK_NATIVE_F16=0", "NK_NATIVE_BF16=0", "NK_DYNAMIC_DISPATCH=1"],
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
                                        "NK_TARGET_HASWELL=1",
                                        "NK_TARGET_SKYLAKE=1",
                                        "NK_TARGET_ICE=1",
                                        "NK_TARGET_GENOA=1",
                                        "NK_TARGET_SAPPHIRE=1",
                                        "NK_TARGET_TURIN=1",
                                        "NK_TARGET_SIERRA=0",
                                        "NK_TARGET_NEON=0",
                                        "NK_TARGET_NEONSDOT=0",
                                        "NK_TARGET_NEONHALF=0",
                                        "NK_TARGET_NEONBFDOT=0",
                                        "NK_TARGET_SVE=0",
                                        "NK_TARGET_SVESDOT=0",
                                        "NK_TARGET_SVEHALF=0",
                                        "NK_TARGET_SVEBFDOT=0",
                                        "NK_TARGET_SVE2=0",
                                    ]
                                },
                            ],
                            [
                                "target_arch=='arm64'",
                                {
                                    "defines": [
                                        "NK_TARGET_HASWELL=0",
                                        "NK_TARGET_SKYLAKE=0",
                                        "NK_TARGET_ICE=0",
                                        "NK_TARGET_GENOA=0",
                                        "NK_TARGET_SAPPHIRE=0",
                                        "NK_TARGET_TURIN=0",
                                        "NK_TARGET_SIERRA=0",
                                        "NK_TARGET_NEON=1",
                                        "NK_TARGET_NEONSDOT=1",
                                        "NK_TARGET_NEONHALF=1",
                                        "NK_TARGET_NEONBFDOT=1",
                                        "NK_TARGET_SVE=1",
                                        "NK_TARGET_SVESDOT=1",
                                        "NK_TARGET_SVEHALF=1",
                                        "NK_TARGET_SVEBFDOT=1",
                                        "NK_TARGET_SVE2=1",
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
                                        "NK_TARGET_HASWELL=1",
                                        "NK_TARGET_SKYLAKE=0",
                                        "NK_TARGET_ICE=0",
                                        "NK_TARGET_GENOA=0",
                                        "NK_TARGET_SAPPHIRE=0",
                                        "NK_TARGET_TURIN=0",
                                        "NK_TARGET_SIERRA=0",
                                        "NK_TARGET_NEON=0",
                                        "NK_TARGET_NEONSDOT=0",
                                        "NK_TARGET_NEONHALF=0",
                                        "NK_TARGET_NEONBFDOT=0",
                                        "NK_TARGET_SVE=0",
                                        "NK_TARGET_SVESDOT=0",
                                        "NK_TARGET_SVEHALF=0",
                                        "NK_TARGET_SVEBFDOT=0",
                                        "NK_TARGET_SVE2=0",
                                    ]
                                },
                            ],
                            [
                                "target_arch=='arm64'",
                                {
                                    "defines": [
                                        "NK_TARGET_HASWELL=0",
                                        "NK_TARGET_SKYLAKE=0",
                                        "NK_TARGET_ICE=0",
                                        "NK_TARGET_GENOA=0",
                                        "NK_TARGET_SAPPHIRE=0",
                                        "NK_TARGET_TURIN=0",
                                        "NK_TARGET_SIERRA=0",
                                        "NK_TARGET_NEON=1",
                                        "NK_TARGET_NEONSDOT=1",
                                        "NK_TARGET_NEONHALF=1",
                                        "NK_TARGET_NEONBFDOT=1",
                                        "NK_TARGET_SVE=0",
                                        "NK_TARGET_SVESDOT=0",
                                        "NK_TARGET_SVEHALF=0",
                                        "NK_TARGET_SVEBFDOT=0",
                                        "NK_TARGET_SVE2=0",
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
                                        "NK_TARGET_HASWELL=1",
                                        "NK_TARGET_SKYLAKE=1",
                                        "NK_TARGET_ICE=1",
                                        "NK_TARGET_GENOA=0",
                                        "NK_TARGET_SAPPHIRE=0",
                                        "NK_TARGET_TURIN=0",
                                        "NK_TARGET_SIERRA=0",
                                        "NK_TARGET_NEON=0",
                                        "NK_TARGET_NEONSDOT=0",
                                        "NK_TARGET_NEONHALF=0",
                                        "NK_TARGET_NEONBFDOT=0",
                                        "NK_TARGET_SVE=0",
                                        "NK_TARGET_SVESDOT=0",
                                        "NK_TARGET_SVEHALF=0",
                                        "NK_TARGET_SVEBFDOT=0",
                                        "NK_TARGET_SVE2=0",
                                    ]
                                },
                            ],
                            [
                                "target_arch=='arm64'",
                                {
                                    "defines": [
                                        "NK_TARGET_HASWELL=0",
                                        "NK_TARGET_SKYLAKE=0",
                                        "NK_TARGET_ICE=0",
                                        "NK_TARGET_GENOA=0",
                                        "NK_TARGET_SAPPHIRE=0",
                                        "NK_TARGET_TURIN=0",
                                        "NK_TARGET_SIERRA=0",
                                        "NK_TARGET_NEON=1",
                                        "NK_TARGET_NEONSDOT=1",
                                        "NK_TARGET_NEONHALF=0",
                                        "NK_TARGET_NEONBFDOT=0",
                                        "NK_TARGET_SVE=0",
                                        "NK_TARGET_SVESDOT=0",
                                        "NK_TARGET_SVEHALF=0",
                                        "NK_TARGET_SVEBFDOT=0",
                                        "NK_TARGET_SVE2=0",
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
