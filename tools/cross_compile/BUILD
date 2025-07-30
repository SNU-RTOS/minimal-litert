load(":cc_toolchain_config.bzl", "cc_toolchain_config")
load("@rules_cc//cc/toolchains:cc_toolchain.bzl", "cc_toolchain")

filegroup(name = "empty")

cc_toolchain_config(
    name = "aarch64_toolchain_config",
)

cc_toolchain(
    name = "aarch64_toolchain",
    toolchain_identifier = "aarch64-fsl-linux",
    toolchain_config = ":aarch64_toolchain_config",
    all_files = ":empty",
    compiler_files = ":empty",
    linker_files = ":empty",
    dwp_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
)


toolchain(
    name = "aarch64_toolchain_entry",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
    toolchain = ":aarch64_toolchain",
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)