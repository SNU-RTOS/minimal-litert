#!/bin/bash
# run.sh
set -euo pipefail

# ========================
if [ ! -d "./log" ]; then
    mkdir -p log
fi
if [ ! -d "./bazel-bin" ]; then
    echo "Bazel build directory not found. Please run './build.sh' first."
    exit 1
fi

# ========================
run_verify() {
    local device=$1
    local model=$2
    
    local model_base
    model_base=$(basename "${model%.*}")
    local log_file="./log/output_verify_${device}_${model_base}.log"
    
    local bin="./bazel-bin/minimal-litert/verify/verify_$device"
    
    {
        echo "[INFO] Run verify_${device}"
        echo "Running verification on $device with model $model"
        if [ ! -f "$bin" ]; then
            echo "Binary $bin not found. Please build the project first."
            exit 1
        fi
        $bin $model
        echo "[INFO] Run verify_${device} finished"
    } | tee "$log_file" 2>&1
    
    echo "===================================="
    echo ""
    echo "Results saved to $log_file"
}

run_main() {
    local device=$1
    local model=$2
    local image=$3
    local labels=$4
    
    local model_base
    model_base=$(basename "${model%.*}")
    local log_file="./log/output_main_${device}_${model_base}.log"
    
    local bin="./bazel-bin/minimal-litert/main/main_$device"
    {
        echo "[INFO] Run main_${device}"
        echo "Running main on $device with model $model and image $image"
        if [ ! -f "$bin" ]; then
            echo "Binary $bin not found. Please build the project first."
            exit 1
        fi
        $bin $model $image $labels
        echo "[INFO] Run main_${device} finished"
    } | tee "$log_file" 2>&1
    echo "===================================="
    echo ""
    echo "Results saved to $log_file"
}

run_main_profile() {
    local model=$1
    local image=$2
    local labels=$3
    local num_threads=${4:-4}
    local delegate_type=${5:-"xnnpack"}

    local model_base
    model_base=$(basename "${model%.*}")

    local csv_file="./log/output_main_profile_${model_base}_${delegate_type}_${num_threads}threads.csv"
    local log_file="./log/output_main_profile_${model_base}_${delegate_type}_${num_threads}threads.log"

    local bin="./bazel-bin/minimal-litert/main/main_profile"
    {
        echo "[INFO] Run main_profile"
        echo "Running main profile with model $model and image $image (threads: $num_threads, delegate: $delegate_type)"
        if [ ! -f "$bin" ]; then
            echo "Binary $bin not found. Please build the project first."
            exit 1
        fi
        $bin $model $image $labels $num_threads $delegate_type $csv_file
        echo "[INFO] Run main_profile finished"
    } | tee "$log_file" 2>&1
    echo "===================================="
    echo ""
    echo "Results saved to $log_file"
}

##################### main #####################
# run_verify cpu ./models/mobilenetv3_small.tflite
# run_verify gpu ./models/mobilenetv3_small.tflite
# run_main cpu ./models/mobilenetv3_small.tflite ./images/dog.jpg ./labels.json
# run_main gpu ./models/mobilenetv3_small.tflite ./images/dog.jpg ./labels.json

# Test with XNNPACK delegate
run_main_profile ./models/mobilenetv3_small.tflite ./images/dog.jpg ./labels.json 8 xnnpack

# Test with GPU delegate
# run_main_profile ./models/mobilenetv3_small.tflite ./images/dog.jpg ./labels.json 8 gpu

