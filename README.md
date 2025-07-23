# Minimal LiteRT Project

This project provides a comprehensive example of using LiteRT (Google's successor to TensorFlow Lite) with the Bazel build system.
It demonstrates inference, verification, and performance profiling with both XNNPACK and GPU delegate support.

> **Note**: This project uses TensorFlow Lite API in Google LiteRT. It is **not compatible** with the new LiteRT-Next API that is currently being developed as the mainline successor. This implementation is based on the current stable LiteRT framework that maintains TensorFlow Lite API compatibility.


## Features

- **LiteRT Integration**: Uses the latest LiteRT framework with modern C API
- **Bazel Build System**: Fully configured with modern Bazel for scalable builds
- **GPU Delegate Support**: XNNPACK and GPU delegate acceleration
- **Performance Profiling**: Comprehensive timing and delegate performance analysis
- **Automated Build/Run Scripts**: Simple `build.sh` and `run.sh` for easy usage
- **Cross-platform**: Optimized for Linux x86_64 with extensible architecture

## Prerequisites

- Bazel 7.1.2 or higher
- Linux environment (Ubuntu/Debian recommended)
- Python 3.10+ (for dependencies)
- Clang compiler (recommended)
- OpenCV 4.x (for image processing)
- JsonCpp (for JSON handling)

## Quick Start

1. Clone the repository:

```bash
git clone <repository-url>
cd minimal-litert
```

2. Run the configure script:

```bash
./configure
```

3. Build the project:

```bash
./build.sh
```

4. Run profiling tests:

```bash
./run.sh
```

## Project Structure

```text
minimal-litert/
├── WORKSPACE              # Bazel workspace configuration
├── .bazelrc              # Bazel build configuration
├── configure             # Build environment setup script
├── configure.py          # Python configuration helper
├── build.sh              # Automated build script
├── run.sh                # Automated run script
├── log/                  # Performance test results
│   └── intel-i9_14900-rtx4070_mobile/  # Hardware-specific results
├── images/               # Test images
├── models/               # TensorFlow Lite models
├── scripts/              # Helper scripts
│   ├── build-benchmark_util.sh
│   ├── common.sh
│   └── install_prerequisites.sh
├── minimal-litert/       # Main source code
│   ├── minimal/          # Basic inference examples
│   ├── main/            # Main implementations
│   │   ├── main_cpu.cpp     # CPU inference
│   │   ├── main_gpu.cpp     # GPU inference
│   │   ├── main_profile.cpp # Performance profiling
│   │   ├── util.cpp         # Utility functions
│   │   └── util.hpp         # Utility headers
│   └── verify/          # Verification tools
│       ├── verify_cpu.cpp  # CPU verification
│       └── verify_gpu.cpp  # GPU verification
└── README.md
```

## Building

### Using Build Script (Recommended)

```bash
# Build all targets
./build.sh

# Build specific configuration
./build.sh --config=debug
./build.sh --config=release
./build.sh --gpu-delegate
```

### Manual Bazel Build

```bash
# Basic build
bazel build //minimal-litert/src:main_cpu_profile

# With GPU delegate
bazel build //minimal-litert/src:main_cpu_profile --config=gpu

# Debug build
bazel build //minimal-litert/src:main_cpu_profile --config=dbg
```

## Running

### Using Run Script (Recommended)

```bash
# Default profiling test
./run.sh

```

### Manual Execution

```bash
# CPU profiling with XNNPACK
./bazel-bin/minimal-litert/main/main_profile \
  ./models/mobilenetv3_small.tflite \
  ./images/dog.jpg \
  ./labels.json \
  8 xnnpack output.csv

# GPU profiling
./bazel-bin/minimal-litert/main/main_profile \
  ./models/mobilenetv3_small.tflite \
  ./images/dog.jpg \
  ./labels.json \
  4 gpu output.csv
```

## Available Build Configurations

| Config         | Description                                           |
| -------------- | ----------------------------------------------------- |
| `linux`        | Basic Linux build with C++17 and warnings suppression |
| `linux_x64`    | Linux build optimized for x86_64 architecture         |
| `avx_linux`    | Linux build with AVX instruction set support          |
| `avx2_linux`   | Linux build with AVX2 and FMA instruction sets        |
| `gpu`          | Enable GPU delegate support                           |
| `dbg`          | Debug build with debug symbols                        |
| `short_logs`   | Build with minimal logging (default)                  |
| `verbose_logs` | Build with verbose compiler output                    |

## Performance Profiling

The project includes comprehensive profiling capabilities:

### Profiling Features

- **Microsecond Precision**: High-resolution timing for accurate performance measurement
- **Delegate Comparison**: Compare XNNPACK vs GPU delegate performance
- **Thread Configuration**: Configurable thread count for CPU inference
- **Detailed Logging**: Comprehensive logs saved to `./log/` directory

### Profiling Output

```text
=== Performance Profile ===
Model: mobilenetv3_small.tflite
Delegate: XNNPACK
Threads: 8
Interpreter Creation: 1234 μs
Model Loading: 2345 μs
Input Setup: 123 μs
Inference: 3456 μs
Output Processing: 234 μs
Total: 7392 μs
```

## GPU Delegate Support

### Requirements

- Compatible GPU with OpenGL ES 3.1+ or OpenCL 1.2+
- Proper GPU drivers installed

### Usage

```bash
# Build with GPU support
./build.sh --gpu-delegate

# Run GPU inference
./run.sh verify --device gpu
./run.sh profile --delegate gpu --threads 4
```

## Customization

### Adding New Inference Examples

1. Create new source file in `minimal-litert/src/`
2. Update `minimal-litert/src/BUILD` file:

```python
cc_binary(
    name = "my_custom_inference",
    srcs = ["my_custom_inference.cpp"],
    deps = [
        "@litert//tflite/c:c_api",
        "@litert//tflite/kernels:builtin_ops",
        ":util",
    ],
)
```

### Extending Profiling

Modify `main_cpu_profile.cpp` to add:

- Custom metrics collection
- Memory usage tracking
- Custom delegate configurations
- Additional output formats

## Dependencies

This project uses the current stable Google LiteRT framework with TensorFlow Lite API compatibility. Automatically managed through Bazel:

- **LiteRT**: Core inference runtime (TensorFlow Lite API compatible)
- **XNNPACK**: Neural network acceleration
- **GPU Delegate**: GPU acceleration support
- **OpenCV**: Image processing (system package)
- **JsonCpp**: JSON handling (system package)
- **Eigen**: Linear algebra library

> **Important**: This project is **not compatible** with LiteRT-Next, which is the future mainline API. Migration to LiteRT-Next will require significant code changes when it becomes stable.

## Troubleshooting

### Build Issues

1. **Missing system libraries**: Install OpenCV and JsonCpp

```bash
sudo apt-get install libopencv-dev libjsoncpp-dev
```

2. **GPU delegate build fails**: Ensure GPU drivers are installed
3. **Bazel cache issues**: Clean and rebuild

```bash
bazel clean --expunge
./build.sh
```

### Runtime Issues

1. **GPU delegate not working**: Check GPU compatibility and drivers
2. **Performance issues**: Try different thread counts and delegates
3. **Model loading fails**: Verify model path and format

## Performance Tips

1. **Use appropriate delegates**:
   - XNNPACK for CPU optimization
   - GPU delegate for compatible hardware
2. **Optimize thread count**: Usually 4-8 threads for best performance
3. **Use release builds**: Significant performance improvement over debug
4. **Profile regularly**: Monitor performance changes with different configurations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes with appropriate tests
4. Update documentation
5. Submit a pull request

## License

This project follows the same license as the original TensorFlow/LiteRT project.
