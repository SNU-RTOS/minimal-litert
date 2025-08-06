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
    
    local bin="./bin/verify_$device"
    
    {
        echo "[INFO] Run verify_${device}"
        echo "Running verification on $device with model $model"
        if [ ! -f "$bin" ]; then
            echo "Binary $bin not found. Please build the project first."
            exit 1
        fi
        $bin $model
        echo "[INFO] Run verify_${device} finished"
    } 2>&1 | tee "$log_file"

    echo "===================================="
    echo ""
    echo "Results saved to $log_file"
}

run_dump_model() {
    local device=$1
    local model=$2
    
    local model_base
    model_base=$(basename "${model%.*}")
    local log_file="./log/${model_base}_dump.log"
    
    local bin="./bin/dump_model_$device"
    if [ ! -f "$bin" ]; then
        echo "Binary $bin not found. Please build the project first."
        exit 1
    fi
    $bin $model $log_file
    
}

run_main() {
    local device=$1
    local model=$2
    local image=$3
    local labels=$4
    
    local model_base
    model_base=$(basename "${model%.*}")
    local log_file="./log/output_main_${device}_${model_base}.log"
    
    local bin="./bin/main_$device"
    {
        echo "[INFO] Run main_${device}"
        echo "Running main on $device with model $model and image $image"
        if [ ! -f "$bin" ]; then
            echo "Binary $bin not found. Please build the project first."
            exit 1
        fi
        $bin $model $image $labels
        echo "[INFO] Run main_${device} finished"
    } 2>&1 | tee "$log_file" 
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
    local warmup_runs=${6:-}
    local profiling_runs=${7:-}
    
    local model_base
    model_base=$(basename "${model%.*}")
    
    local log_file="./log/output_main_profile_${model_base}_${delegate_type}_${num_threads}threads.log"
    local csv_file="./log/output_main_profile_${model_base}_${delegate_type}_${num_threads}threads.csv"
    local tmp_fixed_csv_file="${csv_file}.fixed.csv"
    
    local bin="./bin/main_profile_cpu_only"
    
    echo "[INFO] Run main_profile"
    echo "===================================="
    {
        echo "Running main profile with model $model and image $image (threads: $num_threads, delegate: $delegate_type, warmup: ${warmup_runs:-default}, profiling: ${profiling_runs:-default})"
        
        if [ ! -f "$model" ]; then
            echo "Model file $model not found."
            exit 1
        fi
        if [ ! -f "$image" ]; then
            echo "Image file $image not found."
            exit 1
        fi
        if [ ! -f "$labels" ]; then
            echo "Labels file $labels not found."
            exit 1
        fi
        
        if [ ! -f "$bin" ]; then
            echo "Binary $bin not found. Please build the project first."
            exit 1
        fi
        
        if [[ -n "$warmup_runs" && -n "$profiling_runs" ]]; then
            taskset -c 0-15 $bin $model $image $labels $num_threads $delegate_type $csv_file $warmup_runs $profiling_runs
        else
            taskset -c 0-15 $bin $model $image $labels $num_threads $delegate_type $csv_file
        fi
    } 2>&1 | tee "$log_file"
    echo "===================================="
    echo "[INFO] Run main_profile finished"
    echo "[INFO] Post-processing CSV file..."
    python3 tools/fix_profile_report.py "$csv_file" "$tmp_fixed_csv_file"
    if [ $? -eq 0 ]; then
        mv "$tmp_fixed_csv_file" "$csv_file"
        echo "[INFO] CSV file overwritten with fixed version: $csv_file"
    else
        echo "[ERROR] Failed to fix CSV file. Keeping original."
        rm -f "$tmp_fixed_csv_file"
    fi
    echo ""
    echo "[INFO] Results saved to:"
    echo "  Log : $log_file"
    echo "  CSV : $csv_file"
    
}

##################### main #####################
# run_verify cpu ./models/mobileone_s0.tflite
# run_verify gpu ./models/mobileone_s0.tflite

# run_dump_model cpu ./models/mobileone_s0.tflite

# run_main cpu ./models/mobileone_s0.tflite ./images/dog.jpg ./labels.json
# run_main gpu ./models/mobileone_s0.tflite ./images/dog.jpg ./labels.json

# Test with XNNPACK delegate
# run_main_profile ./models/mobileone_s0.tflite ./images/dog.jpg ./labels.json 12 xnnpack 100 500
run_main_profile ./models/mobilevit_s.cvnets_in1k.tflite ./images/dog.jpg ./labels.json 8 xnnpack 50 100
# Test with GPU delegate
# run_main_profile ./models/mobileone_s0.tflite ./images/dog.jpg ./labels.json 8 gpu 10 10

