workspace(name = "minimal-tflite")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

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

local_repository(
    name = "tensorflow",
    path = "/home/rtos/workspace/minimal-litert/external/tensorflow", # Automatically modified to path to current directory when running setup.sh
)

load("@tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()
load("@tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

