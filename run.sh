#!/bin/bash
# run.sh
set -euo pipefail

# ========================
if [ ! -d "./log" ]; then
    mkdir log
fi
if [ ! -d "./bazel-bin" ]; then
    echo "Bazel build directory not found. Please run 'bazel build //...' first."
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

##################### main #####################
run_verify cpu ./models/mobilenetv3_small.tflite
run_verify gpu ./models/mobilenetv3_small.tflite
run_main cpu ./models/mobilenetv3_small.tflite ./images/dog.jpg ./labels.json
run_main gpu ./models/mobilenetv3_small.tflite ./images/dog.jpg ./labels.json