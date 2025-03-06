#!/bin/bash
source .env

BINARY_NAME=minimal
BINARY_PATH=${ROOT_PATH}/bazel-bin/minimal-tflite/minimal/${BINARY_NAME}


########## Build ##########
bazel build -c opt //minimal-tflite/minimal:minimal
# bazel build -c opt --copt=-DTFLITE_MMAP_DISABLED //ai_edge_torch/generative/examples/cpp:text_generator_main 

########## Make soft symlink ##########
echo "[INFO] Succefully built ${BINARY_NAME}"
echo "[INFO] Making soft symbolic link ${BINARY_NAME} from ${BINARY_PATH} to ${ROOT_PATH}"
if [ "${BINARY_NAME}" ]; then
    rm ${BINARY_NAME}
fi
ln -s ${BINARY_PATH} ${BINARY_NAME}

echo "[INFO] Setup finished."