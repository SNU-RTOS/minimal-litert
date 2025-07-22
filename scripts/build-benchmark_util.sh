#!/bin/bash

# ──────────────────────────────────────────────────────────────────────────────
source common.sh
cd ..
source .env

# ── Build Configuration ───────────────────────────────────────────────────────
BUILD_MODE=${1:-release}
setup_build_config "$BUILD_MODE"

# ── Paths ─────────────────────────────────────────────────────────────────────
BENCHMARK_TOOL_PATH=${LITERT_PATH}/bazel-bin/tflite/tools/benchmark/benchmark_model
BENCHMARK_DEST_PATH=${ROOT_PATH}/bin/benchmark_model

# ── Clean existing binary ─────────────────────────────────────────────────────
if [ -f "${BENCHMARK_DEST_PATH}" ]; then
    rm "${BENCHMARK_DEST_PATH}"
fi

# ── Build LiteRT benchmark tool ───────────────────────────────────────────────
echo "[INFO] Build LiteRT benchmark tool ($BUILD_MODE mode) with GPU delegate support…"
echo "[INFO] Path: ${BENCHMARK_TOOL_PATH}"
echo "[INFO] GPU Flags: ${GPU_FLAGS}"

cd "${LITERT_PATH}" || exit 1
pwd

bazel build ${BAZEL_CONF} \
    //tflite/tools/benchmark:benchmark_model \
    ${GPU_FLAGS} \
    ${COPT_FLAGS} \
    ${GPU_COPT_FLAGS} \
    ${LINKOPTS} \
    --verbose_failures 

# ── Copy binary ───────────────────────────────────────────────────────────────
echo "[INFO] Copy benchmark tool to project root…"
if [ ! -d "${ROOT_PATH}/bin" ]; then
    mkdir -p "${ROOT_PATH}/bin"
fi
cp "${BENCHMARK_TOOL_PATH}" "${BENCHMARK_DEST_PATH}"

if [ -f "${BENCHMARK_DEST_PATH}" ]; then
    echo "✅ Successfully built benchmark tool with GPU delegate support: ${BENCHMARK_DEST_PATH}"
    echo "[INFO] GPU delegate and invoke loop support enabled"
    echo "[INFO] Available flags: --use_gpu, --gpu_invoke_loop_times, --gpu_backend"
else
    echo "❌ Failed to build benchmark tool"
    exit 1
fi

cd "${ROOT_PATH}/scripts"
pwd