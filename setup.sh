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

########## Update Path of tensorflow ##########
WORKSPACE_FILE="${ROOT_PATH}/WORKSPACE"
echo "[INFO] Updating tensorflow local_repository path in ${TENSORFLOW_PATH}/WORKSPACE"
echo "[INFO] BEFOR: $(grep -oE "path = \".*\"" $WORKSPACE_FILE)"
sed -i "s|path = \".*\"|path = \"${TENSORFLOW_PATH}\"|" "$WORKSPACE_FILE"
echo "[INFO] AFTER: $(grep -oE "path = \".*\"" $WORKSPACE_FILE)"
