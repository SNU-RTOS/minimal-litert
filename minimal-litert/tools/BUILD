load("@litert//tflite:build_def.bzl", "tflite_linkopts")
load("@rules_cc//cc:defs.bzl", "cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["@litert//tflite:license"],
    default_visibility = ["//visibility:public"],    
)

cc_binary(
    name = "dump_model",
    srcs = [
        "dump_model_cpu.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "//conditions:default": [
            "-lm",  # math library
        ],
    }),
    deps = [
        "@litert//tflite:framework",
        "@litert//tflite/core:cc_api_stable",
        "@litert//tflite/kernels:builtin_ops",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate",
        "@litert//tflite/profiling:profiler",

    ],
)
