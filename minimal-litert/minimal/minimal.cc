/* Copyright 2024 The LiteRT Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>
#include "tflite/core/c/c_api.h"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// Usage: minimal <tflite model>

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

void PrintInterpreterState(TfLiteInterpreter* interpreter) {
    // TODO: Implement this function to print interpreter state.
    // This is a placeholder.
    printf("Interpreter state printing is not yet implemented.\n");
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];

  // Load model
  TfLiteModel* model = TfLiteModelCreateFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  // TODO: Add options if needed, for example, to add delegates.
  
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(TfLiteInterpreterAllocateTensors(interpreter) == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  PrintInterpreterState(interpreter);

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = (T*)TfLiteInterpreterGetInputTensor(interpreter, i)->data.raw;`

  // Run inference
  TFLITE_MINIMAL_CHECK(TfLiteInterpreterInvoke(interpreter) == kTfLiteOk);
  printf("\n\n=== Post-invoke Interpreter State ===\n");
  PrintInterpreterState(interpreter);

  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = (T*)TfLiteInterpreterGetOutputTensor(interpreter, i)->data.raw;`

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);

  return 0;
}
