#!/bin/bash

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

python3 parser.py -m ../../models/mobileone_s0.tflite
python3 parser.py -m ../../models/mobilenetv3_small.tflite



