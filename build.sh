#!/bin/bash
source .env
bazel build -c opt //minimal-tflite/minimal:minimal

# bazel build -c opt --copt=-DTFLITE_MMAP_DISABLED //ai_edge_torch/generative/examples/cpp:text_generator_main 
