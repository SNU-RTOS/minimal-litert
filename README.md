# Minimal LiteRT Project

This project provides a minimal, customizable example of using LiteRT (Google's successor to TensorFlow Lite) with the Bazel build system. It demonstrates how to build and run inference using LiteRT's C API.

## Features

- **LiteRT Integration**: Uses the latest LiteRT framework instead of legacy TensorFlow Lite
- **Bazel Build System**: Fully configured with modern Bazel for scalable builds
- **Linux Optimized**: Includes optimizations for x86_64 architecture
- **Customizable**: Easy to extend for your own inference needs

## Prerequisites

- Bazel 7.1.2 or higher
- Linux environment (Ubuntu/Debian recommended)
- Python 3.10+ (for dependencies)
- Clang compiler (recommended)

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd minimal-litert
```

2. Run the configure script to set up the build environment:

```bash
./configure
```

## Building

### Basic Build

```bash
bazel build //minimal-litert/minimal:minimal
```

### Optimized Builds

For better performance, you can use various optimization configurations:

#### Linux with generic optimizations:

```bash
bazel build //minimal-litert/minimal:minimal --config=linux
```

#### Linux x86_64 optimizations:

```bash
bazel build //minimal-litert/minimal:minimal --config=linux_x64
```

#### AVX instructions (for compatible CPUs):

```bash
bazel build //minimal-litert/minimal:minimal --config=avx_linux
```

#### AVX2 + FMA instructions (for modern CPUs):

```bash
bazel build //minimal-litert/minimal:minimal --config=avx2_linux
```

#### Debug build:

```bash
bazel build //minimal-litert/minimal:minimal --config=dbg
```

## Running

After building, you can run the minimal inference example:

```bash
./bazel-bin/minimal-litert/minimal/minimal <path-to-your-tflite-model>
```

The program expects a TensorFlow Lite model file as input.

## Available Build Configurations

| Config         | Description                                           |
| -------------- | ----------------------------------------------------- |
| `linux`        | Basic Linux build with C++17 and warnings suppression |
| `linux_x64`    | Linux build optimized for x86_64 architecture         |
| `avx_linux`    | Linux build with AVX instruction set support          |
| `avx2_linux`   | Linux build with AVX2 and FMA instruction sets        |
| `dbg`          | Debug build with debug symbols                        |
| `short_logs`   | Build with minimal logging (default)                  |
| `verbose_logs` | Build with verbose compiler output                    |

## Project Structure

```
minimal-litert/
├── WORKSPACE              # Bazel workspace configuration
├── .bazelrc              # Bazel build configuration
├── configure             # Build environment setup script
├── minimal-litert/
│   └── minimal/
│       ├── BUILD         # Bazel build targets
│       └── minimal.cc    # Main inference example
└── README.md            # This file
```

## Customization

### Adding New Targets

To add new inference examples, modify `minimal-litert/minimal/BUILD`:

```python
cc_binary(
    name = "my_custom_inference",
    srcs = ["my_custom_inference.cc"],
    deps = [
        "@litert//tflite/c:c_api",
        "@litert//tflite/kernels:builtin_ops",
    ],
)
```

### Extending the C++ Code

The main inference code is in `minimal-litert/minimal/minimal.cc`. You can:

1. Add preprocessing/postprocessing logic
2. Support different input/output types
3. Add performance monitoring
4. Implement custom operators

### Build Configuration

Modify `.bazelrc` to add new build configurations or adjust existing ones:

```bash
# Custom optimization for your hardware
build:my_config --copt=-mtune=native
build:my_config --copt=-march=native
build:my_config --copt=-O3
```

## Dependencies

This project automatically manages the following dependencies through Bazel:

- **LiteRT**: Core inference runtime
- **TensorFlow**: Required for LiteRT build system
- **XNNPACK**: Neural network acceleration library
- **Eigen**: Linear algebra library
- **Protocol Buffers**: Serialization library

## Troubleshooting

### Build Errors

1. **Python environment issues**: Run `./configure` to set up the correct Python environment
2. **Missing dependencies**: The WORKSPACE file should handle all dependencies automatically
3. **Compiler errors**: Try using different optimization levels or the `dbg` config

### Performance Issues

1. Use architecture-specific builds (`linux_x64`, `avx2_linux`)
2. Enable XNNPACK optimizations (enabled by default)
3. Use release builds (default `-c opt`)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Test with multiple build configurations
5. Submit a pull request

## License

This project follows the same license as the original TensorFlow/LiteRT project.

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

### 2. Setup repository

(Only needs to be done once)  
Run the following command to setup the repository to build the minimal example:

```sh
./setup
```

### 3. Build binary with Bazel

Run the following command to build the minimal example:

```sh
bazel build -c opt //minimal-tflite/minimal:minimal
ln -s ./bazel/bin/minimal-tflite/minimal/minimal
```

or just

```sh
./build

```

### 4. Run Minimal Binary

Execute the binary file located at `./bazel-bin/minimal-tflite/minimal/minimal`:

```sh
./minimal ./model/yolov5n_float32.tflite
```

or just

```sh
./run
```

## Build With CMAKE

See `./minimal-tflite/minimal-tflite/minimal/README.md` for more details
