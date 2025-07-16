load("@litert//tflite:build_def.bzl", "tflite_linkopts")
load("@rules_cc//cc:defs.bzl", "cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["@litert//tflite:license"],
    default_visibility = ["//visibility:public"],    
)

cc_binary(
    name = "verify_cpu",
    srcs = [
        "verify_cpu.cpp",
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
    ],
)

cc_binary(
    name = "verify_gpu",
    srcs = [
        "verify_gpu.cpp",
    ],
    linkopts = tflite_linkopts() + select({
        "//conditions:default": [
            "-Wl,-rpath,$$ORIGIN/external/litert/bazel-bin/tflite/delegates/gpu",
            "-lEGL",  # EGL 
            "-lGLESv2", # OpenGL ES 2.0 
            "-lm",  # math library
        ],
    }),
    deps = [
        "@litert//tflite:framework",
        "@litert//tflite/core:cc_api_stable",
        "@litert//tflite/kernels:builtin_ops",
        "@litert//tflite/delegates/gpu:delegate",
    ],
)
