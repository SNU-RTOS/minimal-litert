#!/bin/bash

source .env

########## Setup env ##########
BINARY_NAME=minimal
BINARY_PATH=${ROOT_PATH}/bazel-bin/minimal-tflite/minimal/${BINARY_NAME}

echo "[INFO] ROOT_PATH: ${ROOT_PATH}"
echo "[INFO] EXTERNAL PATH: ${EXTERNAL_PATH}"
echo "[INFO] TENSORFLOW_PATH: ${TENSORFLOW_PATH}"

if [ ! -d ${EXTERNAL_PATH} ]; then
    mkdir -p ${EXTERNAL_PATH}
fi

########## Setup external sources ##########
cd ${EXTERNAL_PATH}

## Clone tensorflow
echo "[INFO] Installing tensorflow"
if [ ! -d "./tensorflow" ]; then
    git clone https://github.com/tensorflow/tensorflow.git
    
else
    echo "[INFO] tensorflow is already installed, skipping ..."
fi

WORKSPACE_FILE="${ROOT_PATH}/WORKSPACE"
    sed -i "s|path = \".*\"|path = \"${TENSORFLOW_PATH}\"|" "$WORKSPACE_FILE"
    echo "[INFO] Updated tensorflow local_repository path in ${TENSORFLOW_PATH}/WORKSPACE to: ${TENSORFLOW_PATH}"
