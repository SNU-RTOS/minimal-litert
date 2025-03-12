workspace(name = "minimal-tflite")


load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

local_repository(
    name = "org_tensorflow",
    path = "/root/ghpark/minimal-litert/external/tensorflow", # Automatically modified to path to current directory when running setup.sh
)

new_local_repository(
    name = "tensorflowlite_gpu",
    path = "/root/ghpark/minimal-litert/external/tensorflow/bazel-bin/tensorflow/lite/delegates/gpu",  
    build_file = "@//bazel:tensorflowlite_gpu.BUILD",
)


http_archive(
    name = "bazel_skylib",
    sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
    ],
)
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check(minimum_bazel_version = "3.7.2")

#### Python

# http_archive(
#     name = "rules_python",
#     sha256 = "2ef40fdcd797e07f0b6abda446d1d84e2d9570d234fddf8fcd2aa262da852d1c",
#     strip_prefix = "rules_python-1.2.0",
#     url = "https://github.com/bazelbuild/rules_python/releases/download/1.2.0/rules_python-1.2.0.tar.gz",
# )

# load("@rules_python//python:repositories.bzl", "py_repositories")

# py_repositories()

####

# ABSL on 2023-10-18
http_archive(
    name = "com_google_absl",
    urls = ["https://github.com/abseil/abseil-cpp/archive//9687a8ea750bfcddf790372093245a1d041b21a3.tar.gz"],
    strip_prefix = "abseil-cpp-9687a8ea750bfcddf790372093245a1d041b21a3",
    sha256 = "f841f78243f179326f2a80b719f2887c38fe226d288ecdc46e2aa091e6aa43bc",
)

# XNNPACK on 2024-07-16
http_archive(
    name = "XNNPACK",
    # `curl -L <url> | shasum -a 256`
    sha256 = "0e5d5c16686beff813e3946b26ca412f28acaf611228d20728ffb6479264fe19",
    strip_prefix = "XNNPACK-9ddeb74f9f6866174d61888947e4aa9ffe963b1b",
    url = "https://github.com/google/XNNPACK/archive/9ddeb74f9f6866174d61888947e4aa9ffe963b1b.zip",
)

# Needed by TensorFlow
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "e0a111000aeed2051f29fcc7a3f83be3ad8c6c93c186e64beb1ad313f0c7f9f9",
    strip_prefix = "rules_closure-cf1e44edb908e9616030cc83d085989b8e6cd6df",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",  # 2019-04-04
    ],
)


load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()


#### APPLE 

# http_archive(
#     name = "build_bazel_apple_support",
#     sha256 = "dca96682317cc7112e6fae87332e13a8fefbc232354c2939b11b3e06c09e5949",
#     url = "https://github.com/bazelbuild/apple_support/releases/download/1.19.0/apple_support.1.19.0.tar.gz",
# )

# http_archive(
#     name = "build_bazel_rules_swift",
#     sha256 = "15f7096b41154393da81594909e2db3f5828a5e671b8d873c35788d82f9c97d2",
#     url = "https://github.com/bazelbuild/rules_swift/releases/download/2.7.0/rules_swift.2.7.0.tar.gz",
# )


# http_archive(
#     name = "build_bazel_rules_apple",
#     sha256 = "7d10bbf8ec7bf5d6542122babbb3464e643e981d01967b4d600af392b868d817",
#     url = "https://github.com/bazelbuild/rules_apple/releases/download/3.19.1/rules_apple.3.19.1.tar.gz",
# )

# load(
#     "@build_bazel_rules_apple//apple:repositories.bzl",
#     "apple_rules_dependencies",
# )

# apple_rules_dependencies()

# load(
#     "@build_bazel_rules_swift//swift:repositories.bzl",
#     "swift_rules_dependencies",
# )

# swift_rules_dependencies()

# load(
#     "@build_bazel_rules_swift//swift:extras.bzl",
#     "swift_rules_extra_dependencies",
# )

# swift_rules_extra_dependencies()

# load(
#     "@build_bazel_apple_support//lib:repositories.bzl",
#     "apple_support_dependencies",
# )

# apple_support_dependencies()


####
load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()
# load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
# tf_workspace1()
# load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
# tf_workspace0()


