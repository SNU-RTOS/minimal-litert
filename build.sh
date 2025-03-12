#!/bin/bash
source .env

BINARY_NAME=minimal
BINARY_PATH=${ROOT_PATH}/bazel-bin/minimal-tflite/minimal/${BINARY_NAME}
TFLITE_GPU_DELEGATE_PATH=${TENSORFLOW_PATH}/bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so

########## Build ##########
# bazel build -c opt //minimal-tflite/minimal:minimal
# bazel build -c opt //minimal-tflite/minimal:minimal-with-xnn

if [ !"${TFLITE_GPU_DELEGATE_PATH}" ]; then
    cd ${TENSORFLOW_PATH}
    pwd
    bazel build -c opt \
        --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE \
        --linkopt -s --strip always //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so
    cd ${ROOT_PATH}
    pwd
fi
bazel build -c opt //minimal-tflite/minimal-gpu:minimal-gpu --sandbox_debug
bazel shutdown

########## Make soft symlink ##########
echo "[INFO] Succefully built ${BINARY_NAME}"
echo "[INFO] Making soft symbolic link ${BINARY_NAME} from ${BINARY_PATH} to ${ROOT_PATH}"
if [ "${BINARY_NAME}" ]; then
    rm ${BINARY_NAME}
fi
ln -s ${BINARY_PATH} ${BINARY_NAME}

echo "[INFO] Setup finished."