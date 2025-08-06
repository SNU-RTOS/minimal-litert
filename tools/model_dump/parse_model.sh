#!/bin/bash
# This script parses a TFLite model and generates a schema if necessary.
# This script parses a TFLite model and generates a schema if necessary.
if [ ! -d tflite ]; then
    echo "tflite directory does not exist."
    echo "Generating tflite schema..."
    if arch=$(uname -m); then
        if [ "$arch" = "x86_64" ]; then
            flatc="./flatc_x64"
        fi
        if [ "$arch" = "aarch64" ]; then
            flatc="./flatc_arm64"
        fi
    fi
    $flatc --python schema.fbs
fi



# python3 parser.py -m ../../models/mobileone_s0.tflite
# python3 parser.py -m ../../models/mobilenetv3_small.tflite
# python3 parser.py -m ../../models/mobilenetv3_large_100.ra4_e3600_r224_in1k.tflite
# echo 1 | python3 parser.py -m ../../models/tf_mobilenetv3_large_100.in1k.tflite



# Define the source and destination directories
exported_dir="../../models/exported/"
destination_dir="../../models"

# Find and process .tflite files
# find "$exported_dir" -type f -name "*.tflite" ! -name "*_pt2e_int8.tflite" | while read -r tflite_file; do
#     # Move the .tflite file to the destination directory
#     echo "Moving $tflite_file to $destination_dir"
#     cp "$tflite_file" "$destination_dir/"
# done

# Run parser.py for each .tflite file in the destination directory
for tflite_file in "$destination_dir"/*.tflite; do
    echo "Processing $tflite_file with parser"
    model_name=$(echo "$tflite_file" | grep -oP '[^/]+(?=\.tflite)')
    log="${destination_dir}/${model_name}_dump.log"
    # echo "Running dump_model_cpu_x64 for $tflite_file at $log"
    # echo 1 | python3 parser.py -m "$tflite_file"
    echo 1 | ./dump_model_cpu_x64 "$tflite_file" "$log"
done

