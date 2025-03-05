# TensorFlow Lite C++ Minimal Example

This example demonstrates how to build a simple TensorFlow Lite application.

## Quickstart

### 1. Download Bazelisk

This project requires Bazelisk, a version control tool for Bazel. Bazelisk automatically downloads and runs the appropriate Bazel version based on the `.bazelversion` file in your project.

If Bazelisk is not installed, use one of the following methods to install it:

#### (1) Ubuntu/Linux
```sh
curl -L https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -o /usr/local/bin/bazelisk
chmod +x /usr/local/bin/bazelisk
ln -s /usr/local/bin/bazelisk /usr/local/bin/bazel  # Set up to use 'bazel' command
```

#### (2) macOS (Homebrew)
```sh
brew install bazelisk
```

#### (3) Windows (Scoop)
```sh
scoop install bazelisk
```

#### Downloading a Specific Bazel Version

Bazelisk automatically downloads the Bazel version specified in the `.bazelversion` file in your project's root directory.

- If the `.bazelversion` file is absent, Bazelisk downloads the latest stable version of Bazel.
- If the file is present, Bazelisk downloads and runs the specified version.

### 2. Build Minimal with Bazel

Run the following command to build the minimal example:

```sh
bazel build -c opt //minimal-tflite/minimal:minimal
```

### 3. Run Minimal Binary

Execute the binary file located at `./bazel-bin/minimal-tflite/minimal/minimal`:

```sh
ln -s ./bazel-bin/minimal-tflite/minimal/
./minimal yolov5n_float32.tflite

### minimal <tflite model>
```

## Build With CMAKE
See `./minimal-tflite/minimal-tflite/minimal/README.md` for more details