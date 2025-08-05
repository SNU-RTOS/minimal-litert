workspace(name = "minimal-litert")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Bazel skylib (required for many rules)
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

# Rules shell (from LiteRT official repo)
http_archive(
    name = "rules_shell",
    sha256 = "bc61ef94facc78e20a645726f64756e5e285a045037c7a61f65af2941f4c25e1",
    strip_prefix = "rules_shell-0.4.1",
    url = "https://github.com/bazelbuild/rules_shell/releases/download/v0.4.1/rules_shell-v0.4.1.tar.gz",
)

load("@rules_shell//shell:repositories.bzl", "rules_shell_dependencies", "rules_shell_toolchains")
rules_shell_dependencies()
rules_shell_toolchains()

# Fix abseil-cpp version
http_archive(
    name = "com_google_absl",
    strip_prefix = "abseil-cpp-20240722.1",
    sha256 = "40cee67604060a7c8794d931538cb55f4d444073e556980c88b6c49bb9b19bb7",
    url = "https://github.com/abseil/abseil-cpp/releases/download/20240722.1/abseil-cpp-20240722.1.tar.gz",
)

# Fix protobuf version
http_archive(
    name = "com_google_protobuf",
    strip_prefix = "protobuf-28.3",
    sha256 = "7c3ebd7aaedd86fa5dc479a0fda803f602caaf78d8aff7ce83b89e1b8ae7442a",
    url = "https://github.com/protocolbuffers/protobuf/archive/refs/tags/v28.3.tar.gz",
)


# XNNPACK from local path
local_repository(
    name = "XNNPACK",
    path = "external/XNNPACK",
)

# LiteRT from local path
local_repository(
    name = "litert",
    path = "external/litert",
)

# Load the custom repository rule to select either a local TensorFlow source or a remote http_archive.
load("@litert//litert:tensorflow_source_rules.bzl", "tensorflow_source_repo")

tensorflow_source_repo(
    name = "org_tensorflow",
    sha256 = "",
    strip_prefix = "tensorflow-master",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/master.tar.gz",
    ],
)

# Initialize the TensorFlow repository and all dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()

# Initialize hermetic Python
load("@local_xla//third_party/py:python_init_rules.bzl", "python_init_rules")
python_init_rules()

load("@local_xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")
python_init_repositories(
    default_python_version = "system",
    local_wheel_dist_folder = "dist",
    local_wheel_inclusion_list = [
        "tensorflow*",
        "tf_nightly*",
    ],
    local_wheel_workspaces = ["@org_tensorflow//:WORKSPACE"],
    requirements = {
        "3.9": "@org_tensorflow//:requirements_lock_3_9.txt",
        "3.10": "@org_tensorflow//:requirements_lock_3_10.txt",
        "3.11": "@org_tensorflow//:requirements_lock_3_11.txt",
        "3.12": "@org_tensorflow//:requirements_lock_3_12.txt",
    },
)

load("@local_xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")
python_init_toolchains()

load("@local_xla//third_party/py:python_init_pip.bzl", "python_init_pip")
python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")
install_deps()
# End hermetic Python initialization

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()

load(
    "@local_xla//third_party/py:python_wheel.bzl",
    "python_wheel_version_suffix_repository",
)
python_wheel_version_suffix_repository(name = "tf_wheel_version_suffix")

# CUDA configuration (from LiteRT official repo)
load(
    "@local_xla//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)
cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@local_xla//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@local_xla//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)
cuda_configure(name = "local_config_cuda")

load(
    "@local_xla//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)
nccl_redist_init_repository()

load(
    "@local_xla//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)
nccl_configure(name = "local_config_nccl")

register_toolchains("//tools/cross_compile/toolchain:aarch64_toolchain_entry")