#include <cstdio>
#include <memory>
#include <vector>
#include <cstdlib>

#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/interpreter.h"
#include "tflite/model_builder.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"

#define TFLITE_MINIMAL_CHECK(x)                                     \
    if (!(x))                                                       \
    {                                                               \
        fprintf(stderr, "❌ Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                    \
    }

int main(int argc, char *argv[])
{
    printf("====== verify_cpu ====\n");
    setenv("TF_CPP_MIN_LOG_LEVEL", "0", 1);
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <tflite model>\n", argv[0]);
        return 1;
    }
    const char *filename = argv[1];

    printf("🔍 Loading model from: %s\n", filename);

    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(filename);
    TFLITE_MINIMAL_CHECK(model != nullptr);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    // Apply XNNPACK delegate
    TfLiteXNNPackDelegateOptions xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
    TfLiteDelegate *xnn_delegate = TfLiteXNNPackDelegateCreate(&xnnpack_opts);
    bool delegate_applied = false;

    if (xnn_delegate)
    {
        if (interpreter->ModifyGraphWithDelegate(xnn_delegate) == kTfLiteOk)
        {
            delegate_applied = true;
        }
    }

    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    printf("\n=== Model Summary ===\n");
    printf("📥 Input tensor count  : %zu\n", interpreter->inputs().size());
    printf("📤 Output tensor count : %zu\n", interpreter->outputs().size());
    printf("📦 Total tensor count  : %ld\n", interpreter->tensors_size());
    printf("🔧 Node (op) count     : %zu\n", interpreter->nodes_size());
    printf("🧩 Delegate applied    : %s\n", delegate_applied ? "Yes ✅" : "No ❌");

    if (interpreter->Invoke() == kTfLiteOk)
    {
        printf("🚀 Inference completed successfully.\n");
    }
    else
    {
        printf("❌ Inference failed.\n");
    }

    TfLiteXNNPackDelegateDelete(xnn_delegate);
    return 0;
}
