load("@litert//tflite:build_def.bzl", "tflite_linkopts")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@litert//tflite:license"],
    default_visibility = ["//visibility:public"],
)

# Utility library
cc_library(
    name = "util",
    srcs = ["util.cpp"],
    hdrs = ["util.hpp"],
    copts = [
        "-I/usr/include/opencv4",
        "-I/usr/include/jsoncpp",
    ],
    linkopts = [
        "-ljsoncpp",
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lopencv_imgcodecs",
        "-lopencv_highgui",
    ],
    deps = [
        "@litert//tflite:framework",
        "@litert//tflite/core:cc_api_stable",
        "@litert//tflite/kernels:builtin_ops",
        "@litert//tflite/tools:logging",
    ],
)

# Main CPU inference binary
cc_binary(
    name = "main_cpu",
    srcs = ["main_cpu.cpp"],
    copts = [
        "-I/usr/include/opencv4",
        "-I/usr/include/jsoncpp",
    ],
    linkopts = tflite_linkopts() + [
        "-lm",  # math library
        "-ljsoncpp",
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lopencv_imgcodecs",
        "-lopencv_highgui",
    ],
    deps = [
        ":util",
        "@litert//tflite:framework",
        "@litert//tflite/core:cc_api_stable",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate",
        "@litert//tflite/kernels:builtin_ops",
    ],
)

# Main CPU inference with profiling binary
cc_binary(
    name = "main_profile",
    srcs = ["main_profile.cpp"],
    copts = [
        "-I/usr/include/opencv4",
        "-I/usr/include/jsoncpp",
    ],
    linkopts = tflite_linkopts() + [
        "-Wl,-rpath,$$ORIGIN/external/litert/bazel-bin/tflite/delegates/gpu",
        "-lEGL",  # EGL
        "-lGLESv2",  # OpenGL ES 2.0
        "-lm",  # math library
        "-ljsoncpp",
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lopencv_imgcodecs",
        "-lopencv_highgui",
    ],
    deps = [
        ":util",
        "@litert//tflite:framework",
        "@litert//tflite/core:cc_api_stable",
        "@litert//tflite/kernels:builtin_ops",
        "@litert//tflite/profiling:profile_summarizer",
        "@litert//tflite/profiling:profiler",
        "@litert//tflite/delegates/gpu:delegate",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate",
        "@litert//tflite/tools/benchmark:benchmark_model_lib",
    ],
)

# Main GPU inference binary
cc_binary(
    name = "main_gpu",
    srcs = ["main_gpu.cpp"],
    copts = [
        "-I/usr/include/opencv4",
        "-I/usr/include/jsoncpp",
    ],
    linkopts = tflite_linkopts() + [
        "-Wl,-rpath,$$ORIGIN/external/litert/bazel-bin/tflite/delegates/gpu",
        "-lEGL",  # EGL
        "-lGLESv2",  # OpenGL ES 2.0
        "-lm",  # math library
        "-ljsoncpp",
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lopencv_imgcodecs",
        "-lopencv_highgui",
    ],
    deps = [
        ":util",
        "@litert//tflite:framework",
        "@litert//tflite/core:cc_api_stable",
        "@litert//tflite/delegates/gpu:delegate",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate",
        "@litert//tflite/kernels:builtin_ops",
    ],
)
