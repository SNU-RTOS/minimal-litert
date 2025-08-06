#include <cstdio>
#include <memory>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <iomanip> // for setw, setfill
#include <fstream> // for ofstream

#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/interpreter.h"
#include "tflite/model_builder.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/profiling/profiler.h"
#include "xnnpack/operator.h"

#define TFLITE_MINIMAL_CHECK(x)                                     \
    if (!(x))                                                       \
    {                                                               \
        fprintf(stderr, "❌ Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                    \
    }

namespace
{
    void InspectExecutionPlan(tflite::Interpreter *interpreter,
                              const std::string &subgraph_name = "main",
                              std::ostream *output = nullptr)
    {
        std::ostream &out = (output != nullptr) ? *output : std::cout;

        const auto &execution_plan = interpreter->execution_plan();

        out << "\nSubgraph: " << subgraph_name << " "
            << "(Total Nodes: " << execution_plan.size()
            << ", Including Nested: " << execution_plan.size()
            << ", STABLEHLO_COMPOSITE: 0)"
            << std::endl;

        for (int idx : execution_plan)
        {
            const auto *node_and_reg = interpreter->node_and_registration(idx);
            auto *node = &node_and_reg->first;
            auto *reg = &node_and_reg->second;
            std::string op_name =
                tflite::EnumNameBuiltinOperator(
                    static_cast<tflite::BuiltinOperator>(reg->builtin_code));

            // print with zero-padded index
            out << "  [" << std::setw(4) << std::setfill('0') << idx << "] " << op_name;
            if (reg->builtin_code == tflite::BuiltinOperator_DELEGATE)
            {
                TfLiteXNNPackDelegateInspect(node->user_data, output);
            }
            else
            {
                out << std::endl;
            }
        }
        out << std::endl;
    }

    void InspectSignatureExecutionPlan(tflite::Interpreter *interpreter,
                                       const std::string &signature_key,
                                       std::ostream *output = nullptr)
    {
        std::ostream &out = (output != nullptr) ? *output : std::cout;

        auto runner = interpreter->GetSignatureRunner(signature_key.c_str());
        if (!runner)
        {
            out << "Failed to get signature runner for: " << signature_key << std::endl;
            return;
        }

        // Inspect the execution plan with signature context
        InspectExecutionPlan(interpreter, signature_key, output);

        /*
        // Print signature I/O information
        const auto &input_names = runner->input_names();
        const auto &output_names = runner->output_names();

        out << "  Signature Inputs (" << input_names.size() << "): ";
        for (const auto &name : input_names)
        {
            out << name << " ";
        }
        out << std::endl;

        out << "  Signature Outputs (" << output_names.size() << "): ";
        for (const auto &name : output_names)
        {
            out << name << " ";
        }
        out << std::endl;
        */
    }

    void InspectAllSignatures(tflite::Interpreter *interpreter, std::ostream *output = nullptr)
    {
        const auto &signature_keys = interpreter->signature_keys();

        // Inspect execution plan for each signature
        for (const std::string *key : signature_keys)
        {
            InspectSignatureExecutionPlan(interpreter, *key, output);
        }
    }

    void PrintModelSignature(tflite::Interpreter *interpreter, int &sig_index)
    {
        // Print model signature keys (console only)
        std::cout << "\n=== Model Signature ===" << std::endl;
        sig_index = -1;
        if (interpreter)
        {
            const std::vector<const std::string *> &keys = interpreter->signature_keys();
            std::cout << "The Model contains " << keys.size() << " signature key(s)." << std::endl;
            if (!keys.empty())
            {
                for (int i = 0; i < keys.size(); ++i)
                {
                    const std::string *key = keys[i];
                    std::cout << "  " << std::setfill('0') << i << ": "
                              << *key << std::endl;
                }
            }

            std::cout << "Please select a signature index (0 to " << keys.size() - 1 << "): ";
            std::cin >> sig_index;
        }
        else
        {
            std::cout << "The Model does not contain any signature keys.";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char *argv[])
{
    setenv("TF_CPP_MIN_LOG_LEVEL", "0", 1);
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <tflite model> <log file>\n", argv[0]);
        return 1;
    }
    const char *model_file_name = argv[1];
    const char *log_file_name = argv[2];

    // Generate log file name
    // std::string log_filename = GenerateLogFileName(filename);
    std::ofstream log_file(log_file_name);
    if (!log_file.is_open())
    {
        fprintf(stderr, "❌ Failed to open log file: %s\n", log_file_name);
        return 1;
    }

    std::cout << "====== dump_model_cpu ======" << std::endl;
    std::cout << "🔍 Loading model from: " << model_file_name << std::endl;
    std::cout << "📝 Logging output to: " << log_file_name << std::endl;

    // 1. Load the model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_file_name);
    TFLITE_MINIMAL_CHECK(model != nullptr);

    // 2. Create Op resolver
    tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

    // 3. Create Interpreter
    std::unique_ptr<tflite::Interpreter> interpreter;

    // 3.5 Create a profiler
    auto profiler = std::make_unique<tflite::profiling::Profiler>();

    // 4. Create InterpreterBuilder and Initialize Interpreter
    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&interpreter);

    TFLITE_MINIMAL_CHECK(interpreter != nullptr);
    interpreter->SetProfiler(profiler.get());

    // Print model signature (console only, as requested)
    int sig_index = 0;
    PrintModelSignature(interpreter.get(), sig_index);

    // Inspect all signatures before delegate
    log_file << "\n=== Before Applying Delegate ===" << std::endl;

    InspectAllSignatures(interpreter.get(), &log_file);

    // 5. Apply XNNPACK delegate
    TfLiteXNNPackDelegateOptions xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
    TfLiteDelegate *xnn_delegate = TfLiteXNNPackDelegateCreate(&xnnpack_opts);
    bool delegate_applied = false;

    if (xnn_delegate && interpreter->ModifyGraphWithDelegate(xnn_delegate) == kTfLiteOk)
        delegate_applied = true;

    // 6. Allocate tensors
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    // print model execution plan after applying delegate
    if (delegate_applied)
    {
        log_file << "\n=== After Applying Delegate ===" << std::endl;

        // Inspect all signatures after delegate
        InspectAllSignatures(interpreter.get(), &log_file);
    }

    // 7. Deallocate delegate
    if (delegate_applied)
        TfLiteXNNPackDelegateDelete(xnn_delegate);

    log_file.close();

    std::cout << "\n✔ Parsing complete. Log saved to: " << log_file_name << std::endl;

    return 0;
}