package(
    default_visibility = ["//visibility:public"],
)

cc_binary(
    name = "minimal",
    srcs = [
        "minimal.cc",
    ],
    deps = [
        "@litert//tflite/c:c_api",
        "@litert//tflite/kernels:builtin_ops",
    ],
)

cc_binary(
    name = "minimal-with-xnn",
    srcs = [
        "minimal-with-xnn.cc",
    ],
    deps = [
        "@litert//tflite/c:c_api",
        "@litert//tflite/kernels:builtin_ops",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate",
    ],
)
