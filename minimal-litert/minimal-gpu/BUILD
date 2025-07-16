load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_linkopts")

licenses = ["notice"]

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],    
)


cc_binary(
    name = "minimal-gpu",
    srcs = [
        "minimal-gpu.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "@org_tensorflow//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [
            "-Wl,-rpath,$$ORIGIN/external/tensorflow/bazel-bin/tensorflow/lite/delegates/gpu",
            "-lEGL",  # EGL 
            "-lGLESv2" # OpenGL ES 2.0 
        ],
    }),
    includes = ["@org_tensorflow//tensorflow/lite/delegates/gpu"],
    deps = [
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/core:cc_api_stable",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "@tensorflowlite_gpu//:tensorflowlite_gpu_lib",
        
    ],
)
