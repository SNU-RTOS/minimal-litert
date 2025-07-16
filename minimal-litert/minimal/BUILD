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
