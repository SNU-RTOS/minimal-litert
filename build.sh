#!/bin/bash
# build.sh - Build script for minimal-litert project
set -euo pipefail

# Default configuration
BUILD_MODE="release"
BUILD_TARGET="all"
CLEAN_BUILD=false
VERBOSE=false

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS] [TARGET]

Build script for minimal-litert project

OPTIONS:
    -h, --help          Show this help message
    -d, --debug         Build in debug mode (default: release)
    -r, --release       Build in release mode
    -c, --clean         Clean build (remove bazel cache)
    -v, --verbose       Verbose output
    -t, --target TARGET Specific target to build (default: all)

TARGETS:
    all                 Build all targets
    main                Build main targets only
    verify              Build verify targets only
    profile             Build profile targets only
    
EXAMPLES:
    $0                              # Build all targets in release mode
    $0 --debug                      # Build all targets in debug mode
    $0 --target main                # Build only main targets
    $0 --clean --release            # Clean build in release mode
    $0 --debug --target profile     # Build profile targets in debug mode

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--debug)
                BUILD_MODE="debug"
                shift
                ;;
            -r|--release)
                BUILD_MODE="release"
                shift
                ;;
            -c|--clean)
                CLEAN_BUILD=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -t|--target)
                BUILD_TARGET="$2"
                shift 2
                ;;
            *)
                BUILD_TARGET="$1"
                shift
                ;;
        esac
    done
}

# Setup build configuration
setup_build_config() {
    local BUILD_MODE=${1:-release}
    
    if [ "$BUILD_MODE" = "debug" ]; then
        BAZEL_CONF="-c dbg --config=linux"
        COPT_FLAGS="--copt=-Og"
        LINKOPTS=""
    else
        BAZEL_CONF="-c opt --config=linux"
        COPT_FLAGS="--copt=-Os --copt=-fPIC"
        LINKOPTS="--linkopt=-s"
    fi

    # GPU Delegate Configuration
    GPU_FLAGS="--define=supports_gpu_delegate=true"
    GPU_COPT_FLAGS="--copt=-DTFLITE_GPU_ENABLE_INVOKE_LOOP=1 --copt=-DCL_DELEGATE_NO_GL --copt=-DTFLITE_SUPPORTS_GPU_DELEGATE=1"
    
    # Export variables for use in calling scripts
    export BAZEL_CONF COPT_FLAGS LINKOPTS GPU_FLAGS GPU_COPT_FLAGS
    
    echo "[INFO] Build configuration:"
    echo "  Mode: $BUILD_MODE"
    echo "  Bazel config: $BAZEL_CONF"
    echo "  C++ flags: $COPT_FLAGS"
    echo "  Link flags: $LINKOPTS"
    echo "  GPU flags: $GPU_FLAGS $GPU_COPT_FLAGS"
}

# Clean build function
clean_build() {
    echo "[INFO] Cleaning build cache..."
    bazel clean --expunge
    echo "[INFO] Build cache cleaned"
}

# Build function
build_target() {
    local target=$1
    local build_flags="$BAZEL_CONF $COPT_FLAGS $LINKOPTS $GPU_FLAGS $GPU_COPT_FLAGS"
    
    if [ "$VERBOSE" = true ]; then
        build_flags="$build_flags --verbose_failures"
    fi
    
    echo "[INFO] Building target: $target"
    echo "[INFO] Build flags: $build_flags"
    
    case $target in
        "all")
            echo "[INFO] Building all targets..."
            bazel build $build_flags //minimal-litert/...
            ;;
        "main")
            echo "[INFO] Building main targets..."
            bazel build $build_flags //minimal-litert/main/...
            ;;
        "verify")
            echo "[INFO] Building verify targets..."
            bazel build $build_flags //minimal-litert/verify/...
            ;;
        "profile")
            echo "[INFO] Building profile targets..."
            bazel build $build_flags //minimal-litert/main:main_profile
            ;;
        *)
            echo "[INFO] Building custom target: $target"
            bazel build $build_flags $target
            ;;
    esac
}

# Main execution
main() {
    parse_args "$@"
    
    echo "====== minimal-litert Build Script ======"
    echo "[INFO] Build mode: $BUILD_MODE"
    echo "[INFO] Build target: $BUILD_TARGET"
    echo "[INFO] Clean build: $CLEAN_BUILD"
    echo "[INFO] Verbose: $VERBOSE"
    echo "=========================================="
    
    # Setup build configuration
    setup_build_config "$BUILD_MODE"
    
    # Clean build if requested
    if [ "$CLEAN_BUILD" = true ]; then
        clean_build
    fi
    
    # Build target
    build_target "$BUILD_TARGET"
    
    echo "[INFO] Build completed successfully!"
    echo "[INFO] Built binaries are available in bazel-bin/"
    
    # Show available binaries
    echo ""
    echo "[INFO] Available binaries:"
    if [ -d "bazel-bin/minimal-litert" ]; then
        find bazel-bin/minimal-litert -type f -executable | sort
    else
        echo "  No binaries found. Build may have failed."
    fi
}

# Run main function
main "$@"
