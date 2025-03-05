#!/bin/bash

source .env

########## Setup env ##########
ROOT_PATH=$(pwd)
TENSORFLOW_PATH=${ROOT_PATH}/${TENSORFLOW_PATH}

BINARY_NAME=minimal
BINARY_PATH=${ROOT_PATH}/bazel-bin/minimal-tflite/minimal/${BINARY_NAME}

echo "[INFO] TENSORFLOW_PATH: ${TENSORFLOW_PATH}"

if [ ! -d "./external" ]; then
    mkdir -p ./external
fi

########## Setup external sources ##########
cd external

## Clone tensorflow
echo "[INFO] Installing tensorflow"
if [ ! -d "./tensorflow" ]; then
    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow
else
    echo "[INFO] tensorflow is already installed, skipping ..."
fi

########## Build LiteRT_LLM_Inference_app ##########
echo "[INFO] Build ${BINARY_NAME}"
echo "========================"
cd ${ROOT_PATH}
${ROOT_PATH}/build.sh
cd ${ROOT_PATH}
echo "========================"

