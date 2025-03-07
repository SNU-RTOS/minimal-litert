load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_linkopts")

licenses = ["notice"]

package(
    # copybara:uncomment default_applicable_licenses = ["@tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],    
)

cc_binary(
    name = "minimal",
    srcs = [
        "minimal.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "@org_tensorflow//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/core:cc_api_stable",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)
