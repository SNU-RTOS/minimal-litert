#include <cstdio>
#include <memory>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <iomanip> // for setw, setfill
#include <fstream>  // for ofstream

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
                              std::ostream* output = nullptr)
    {
        std::ostream& out = (output != nullptr) ? *output : std::cout;
        
        const auto &execution_plan = interpreter->execution_plan();
        // Header in requested format
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
            out << "  [" << std::setw(4) << std::setfill('0') << idx
                << "] " << op_name;
            if (reg->builtin_code == tflite::BuiltinOperator_DELEGATE)
            {
                // Note: TfLiteXNNPackDelegateInspect outputs to stdout directly
                out << " (DELEGATE - details in stdout)" << std::endl;
                if (output == nullptr) {
                    TfLiteXNNPackDelegateInspect(node->user_data);
                }
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
                                       std::ostream* output = nullptr)
    {
        std::ostream& out = (output != nullptr) ? *output : std::cout;
        
        auto runner = interpreter->GetSignatureRunner(signature_key.c_str());
        if (!runner)
        {
            out << "Failed to get signature runner for: " << signature_key << std::endl;
            return;
        }

        // Print signature-specific execution plan
        out << "\n=== Execution Plan for Signature: " << signature_key << " ===" << std::endl;

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

    void InspectAllSignatures(tflite::Interpreter *interpreter, std::ostream* output = nullptr)
    {
        std::ostream& out = (output != nullptr) ? *output : std::cout;
        
        const auto &signature_keys = interpreter->signature_keys();

        if (signature_keys.empty())
        {
            out << "\nNo signatures found. Inspecting main execution plan only." << std::endl;
            InspectExecutionPlan(interpreter, "main", output);
            return;
        }

        // Inspect execution plan for each signature
        for (const std::string *key : signature_keys)
        {
            InspectSignatureExecutionPlan(interpreter, *key, output);
        }
    }

    void PrintModelSignature(tflite::Interpreter *interpreter)
    {
        // Print model signature keys (console only)
        std::cout << "\n=== Model Signature ===" << std::endl;

        if (interpreter)
        {
            const std::vector<const std::string *> &keys = interpreter->signature_keys();
            std::cout << "The Model contains " << keys.size() << " signature key(s)." << std::endl;
            if (!keys.empty())
            {
                for (int i = 0; i < keys.size(); ++i)
                {
                    const std::string *key = keys[i];
                    std::cout << "  [" << std::setw(4) << std::setfill('0') << i << "] "
                              << *key << std::endl;
                }
            }
        }
        else
        {
            std::cout << "The Model does not contain any signature keys.";
        }
        std::cout << std::endl;
    }

    std::string GenerateLogFileName(const std::string& model_file)
    {
        // Extract base filename without extension
        size_t last_slash = model_file.find_last_of("/\\");
        size_t last_dot = model_file.find_last_of(".");
        
        std::string base_name;
        if (last_slash == std::string::npos) {
            base_name = (last_dot == std::string::npos) ? model_file : model_file.substr(0, last_dot);
        } else {
            std::string filename = model_file.substr(last_slash + 1);
            base_name = (last_dot == std::string::npos) ? filename : filename.substr(0, last_dot - last_slash - 1);
        }
        
        return base_name + "_dump.log";
    }
}

int main(int argc, char *argv[])
{
    setenv("TF_CPP_MIN_LOG_LEVEL", "0", 1);
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <tflite model>\n", argv[0]);
        return 1;
    }
    const char *filename = argv[1];

    // Generate log file name
    // std::string log_filename = GenerateLogFileName(filename);
    // std::ofstream log_file(log_filename);
    // if (!log_file.is_open())
    // {
    //     fprintf(stderr, "❌ Failed to open log file: %s\n", log_filename.c_str());
    //     return 1;
    // }

    std::cout << "====== dump_model_cpu ======" << std::endl;
    std::cout << "🔍 Loading model from: " << filename << std::endl;
    // std::cout << "📝 Logging output to: " << log_filename << std::endl;

    // 1. Load the model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename);
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
    PrintModelSignature(interpreter.get());

    // Inspect all signatures before delegate
    std::cout << "\n=== Before Delegate Application ===" << std::endl;

    InspectAllSignatures(interpreter.get(), nullptr);

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
        std::cout << "\n=== After Delegate Application ===" << std::endl;
        
        // Inspect all signatures after delegate
        InspectAllSignatures(interpreter.get(), nullptr);
    }

    // 7. Deallocate delegate
    if (delegate_applied)
        TfLiteXNNPackDelegateDelete(xnn_delegate);

    
    // std::cout << "\n✔ Parsing complete. Log saved to: " << log_filename << std::endl;
    
    return 0;
}