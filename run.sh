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
    local logfile="./log/output_verify_${device}_${model_base}.log"
    
    local bin="./bazel-bin/minimal-litert/verify/verify_$device"
    
    {
        exec > >(tee "$logfile") 2>&1
        echo "[INFO] Run verify_${device}"
        echo "Running verification on $device with model $model"
        if [ ! -f "$bin" ]; then
            echo "Binary $bin not found. Please build the project first."
            exit 1
        fi
        $bin $model
        echo "[INFO] Run verify_${device} finished"
        echo "Results saved to $logfile"
        echo ""
    }
}

run_main() {
    local device=$1
    local model=$2
    local image=$3
    local labels=$4
    
    local model_base
    model_base=$(basename "${model%.*}")
    local logfile="./log/output_main_${device}_${model_base}.log"
    
    local bin="./bazel-bin/minimal-litert/main/main_$device"
    {
        exec > >(tee "$logfile") 2>&1
        echo "[INFO] Run main_${device}"
        echo "Running main on $device with model $model and image $image"
        if [ ! -f "$bin" ]; then
            echo "Binary $bin not found. Please build the project first."
            exit 1
        fi
        $bin $model $image $labels
        echo "[INFO] Run main_${device} finished"
        echo "Results saved to $logfile"
        echo ""
    }
}

run_main_profile() {
    local model=$1
    local image=$2
    local labels=$3
    local enable_profile=$4
    local num_threads=${5:-4}  # Default to 4 threads if not specified
    local delegate_type=${6:-"xnnpack"}  # Default to xnnpack if not specified
    
    local model_base
    model_base=$(basename "${model%.*}")
    local logfile="./log/output_main_profile_${model_base}_${delegate_type}_threads${num_threads}.log"
    
    local bin="./bazel-bin/minimal-litert/main/main_profile"
    {
        exec > >(tee "$logfile") 2>&1
        echo "[INFO] Run main_profile"
        echo "Running main profile with model $model and image $image (threads: $num_threads, delegate: $delegate_type)"
        if [ ! -f "$bin" ]; then
            echo "Binary $bin not found. Please build the project first."
            exit 1
        fi
        $bin $model $image $labels $enable_profile $num_threads $delegate_type
        echo "[INFO] Run main_profile finished"
        echo "Results saved to $logfile"
        echo ""
    }
}

##################### main #####################
# run_verify cpu ./models/mobilenetv3_small.tflite
# run_verify gpu ./models/mobilenetv3_small.tflite
# run_main cpu ./models/mobilenetv3_small.tflite ./images/dog.jpg ./labels.json
# run_main gpu ./models/mobilenetv3_small.tflite ./images/dog.jpg ./labels.json

# Test with XNNPACK delegate
run_main_profile ./models/mobilenetv3_small.tflite ./images/dog.jpg ./labels.json 1 8 xnnpack
run_main_profile ./models/mobilenetv3_small.tflite ./images/dog.jpg ./labels.json 1 4 gpu

# Test with GPU delegate
# run_main_profile ./models/mobilenetv3_small.tflite ./images/dog.jpg ./labels.json 1 4 gpu

