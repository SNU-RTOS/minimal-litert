load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl", "feature", "flag_group", "flag_set", "tool_path")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")


all_link_actions = [
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

def _impl(ctx):
    tool_paths = [
        tool_path(
            name = "gcc",
            path = "/opt/fsl-goldvip-no-hv/1.12.0_Auto_Linux_BSP_41.0/sysroots/x86_64-fslbsp-linux/usr/bin/aarch64-fsl-linux/aarch64-fsl-linux-gcc",
        ),
        tool_path(
            name = "ld",
            path = "/opt/fsl-goldvip-no-hv/1.12.0_Auto_Linux_BSP_41.0/sysroots/x86_64-fslbsp-linux/usr/bin/aarch64-fsl-linux/aarch64-fsl-linux-ld",
        ),
        tool_path(
            name = "ar",
            path = "/opt/fsl-goldvip-no-hv/1.12.0_Auto_Linux_BSP_41.0/sysroots/x86_64-fslbsp-linux/usr/bin/aarch64-fsl-linux/aarch64-fsl-linux-ar",
        ),
        tool_path(
            name = "cpp",
            path = "/opt/fsl-goldvip-no-hv/1.12.0_Auto_Linux_BSP_41.0/sysroots/x86_64-fslbsp-linux/usr/bin/aarch64-fsl-linux/aarch64-fsl-linux-cpp",
        ),
        tool_path(
            name = "gcov",
            path = "/opt/fsl-goldvip-no-hv/1.12.0_Auto_Linux_BSP_41.0/sysroots/x86_64-fslbsp-linux/usr/bin/aarch64-fsl-linux/aarch64-fsl-linux-gcov",
        ),
        tool_path(
            name = "nm",
            path = "/opt/fsl-goldvip-no-hv/1.12.0_Auto_Linux_BSP_41.0/sysroots/x86_64-fslbsp-linux/usr/bin/aarch64-fsl-linux/aarch64-fsl-linux-nm",
        ),
        tool_path(
            name = "objdump",
            path = "/opt/fsl-goldvip-no-hv/1.12.0_Auto_Linux_BSP_41.0/sysroots/x86_64-fslbsp-linux/usr/bin/aarch64-fsl-linux/aarch64-fsl-linux-objdump",
        ),
        tool_path(
            name = "strip",
            path = "/opt/fsl-goldvip-no-hv/1.12.0_Auto_Linux_BSP_41.0/sysroots/x86_64-fslbsp-linux/usr/bin/aarch64-fsl-linux/aarch64-fsl-linux-strip",
        ),
        tool_path(
            name = "cxx",
            path = "/opt/fsl-goldvip-no-hv/1.12.0_Auto_Linux_BSP_41.0/sysroots/x86_64-fslbsp-linux/usr/bin/aarch64-fsl-linux/aarch64-fsl-linux-g++",
        ),
    ]

    features = [
        feature(
            name = "default_linker_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = ([
                        flag_group(
                            flags = [
                                "--sysroot=/opt/fsl-goldvip-no-hv/1.12.0_Auto_Linux_BSP_41.0/sysroots/cortexa53-crypto-fsl-linux",
                                "-march=armv8-a",
                                "-mtune=cortex-a53",
                                "-lstdc++",
                            ],
                        ),
                    ]),
                ),
            ],
        ),
        feature(
            name = "default_compile_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.assemble,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "--sysroot=/opt/fsl-goldvip-no-hv/1.12.0_Auto_Linux_BSP_41.0/sysroots/cortexa53-crypto-fsl-linux",
                            ],
                        )
                    ],
                )
            ],
        ),
      
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        tool_paths = tool_paths,
        cxx_builtin_include_directories = [
            "/opt/fsl-goldvip-no-hv/1.12.0_Auto_Linux_BSP_41.0/sysroots/x86_64-fslbsp-linux/usr/lib/aarch64-fsl-linux/gcc/aarch64-fsl-linux/11.4.0/include",
            "/opt/fsl-goldvip-no-hv/1.12.0_Auto_Linux_BSP_41.0/sysroots/x86_64-fslbsp-linux/usr/lib/aarch64-fsl-linux/gcc/aarch64-fsl-linux/11.4.0/include-fixed",  # Add this line
            "/opt/fsl-goldvip-no-hv/1.12.0_Auto_Linux_BSP_41.0/sysroots/cortexa53-crypto-fsl-linux/usr/include",
        ],
        toolchain_identifier = "aarch64-fsl-linux",
        host_system_name = "x86_64",
        target_system_name = "aarch64-fsl-linux",
        target_cpu = "aarch64",
        compiler = "gcc",
        abi_version = "gcc",
        abi_libc_version = "glibc",
        target_libc = "glibc",
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)
