// xnn-delegate-main
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

#include "opencv2/opencv.hpp" //opencv

#include "tflite/delegates/xnnpack/xnnpack_delegate.h" //for xnnpack delegate
#include "tflite/delegates/gpu/delegate.h"             //for gpu delegate

#include "tflite/model_builder.h"
#include "tflite/interpreter_builder.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model.h"
#include "tflite/profiling/profile_summarizer.h"
#include "tflite/profiling/buffered_profiler.h"
#include "tflite/profiling/profile_summary_formatter.h"
#include "tflite/tools/benchmark/benchmark_model.h"
#include "tflite/tools/benchmark/benchmark_params.h"
#include "tflite/tools/logging.h"
#include "util.hpp"

// Configuration constants
static const int WARMUP_RUNS = 5;
static const int PROFILING_RUNS = 1;

// Function declarations
void apply_xnnpack_delegate(std::unique_ptr<tflite::Interpreter> &interpreter,
                            TfLiteDelegate *&delegate, bool &delegate_applied, int num_threads);
void apply_gpu_delegate(std::unique_ptr<tflite::Interpreter> &interpreter,
                        TfLiteDelegate *&delegate, bool &delegate_applied);
void process_input_tensor(std::unique_ptr<tflite::Interpreter> &interpreter,
                          const cv::Mat &preprocessed_image);
void process_output_tensor(std::unique_ptr<tflite::Interpreter> &interpreter,
                           std::vector<float> &probs);
void run_warmup(std::unique_ptr<tflite::Interpreter> &interpreter, TfLiteDelegate *delegate);
void run_inference(std::unique_ptr<tflite::Interpreter> &interpreter, TfLiteDelegate *delegate,
                   std::unique_ptr<tflite::profiling::BufferedProfiler> &profiler, bool enable_profiling);
void print_profiling_results(std::unique_ptr<tflite::profiling::BufferedProfiler> &profiler,
                             std::unique_ptr<tflite::Interpreter> &interpreter, bool enable_profiling);

int main(int argc, char *argv[])
{
    std::cout << "====== main_cpu with profiling ====" << std::endl;

    // Parse command line arguments
    if (argc < 4 || argc > 7)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path> <label_json_path> [enable_profiling=1] [num_threads=4] [delegate_type=xnnpack]" << std::endl;
        std::cerr << "  delegate_type: xnnpack or gpu" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string label_path = argv[3];
    bool enable_profiling = (argc >= 5 && std::string(argv[4]) == "1");
    int num_threads = (argc >= 6) ? std::atoi(argv[5]) : 4;
    std::string delegate_type = (argc == 7) ? std::string(argv[6]) : "xnnpack";

    // Determine model type from filename
    bool is_int8_model = (model_path.find("int8") != std::string::npos);
    std::cout << "[INFO] Model type detected: " << (is_int8_model ? "INT8" : "FP32") << std::endl;
    std::cout << "[INFO] Profiling enabled: " << (enable_profiling ? "YES" : "NO") << std::endl;
    std::cout << "[INFO] Number of threads: " << num_threads << std::endl;
    std::cout << "[INFO] Delegate type: " << delegate_type << std::endl;

    /* Load model */
    util::timer_start("Load Model");
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model)
    {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    util::timer_stop("Load Model");

    /* Build interpreter */
    util::timer_start("Build Interpreter");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    util::timer_stop("Build Interpreter");

    // Setup profiler
    std::unique_ptr<tflite::profiling::BufferedProfiler> profiler;
    if (enable_profiling)
    {
        profiler = std::make_unique<tflite::profiling::BufferedProfiler>(1024);
        interpreter->SetProfiler(profiler.get());
        std::cout << "[INFO] Profiler created (will start after warmup)" << std::endl;
    }

    /* Apply delegate */
    util::timer_start("Apply Delegate");
    TfLiteDelegate *delegate = nullptr;
    bool delegate_applied = false;
    if (delegate_type == "xnnpack")
    {
        apply_xnnpack_delegate(interpreter, delegate, delegate_applied, num_threads);
    }
    else if (delegate_type == "gpu")
    {
        apply_gpu_delegate(interpreter, delegate, delegate_applied);
    }
    else
    {
        std::cerr << "[ERROR] Unknown delegate type: " << delegate_type << std::endl;
        return 1;
    }
    util::timer_stop("Apply Delegate");

    /* Allocate Tensor */
    util::timer_start("Allocate Tensor");
    if (!interpreter || interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to initialize interpreter" << std::endl;
        return 1;
    }
    util::timer_stop("Allocate Tensor");

    util::print_model_summary(interpreter.get(), delegate_applied);

    /* Load input image */
    util::timer_start("Load Input Image");
    cv::Mat origin_image = cv::imread(image_path);
    if (origin_image.empty())
    {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    util::timer_stop("Load Input Image");

    /* Preprocessing */
    util::timer_start("E2E Total(Pre+Inf+Post)");
    util::timer_start("Preprocessing");

    // Get input tensor info
    TfLiteTensor *input_tensor = interpreter->input_tensor(0);
    int input_height = input_tensor->dims->data[1];
    int input_width = input_tensor->dims->data[2];

    std::cout << "\n[INFO] Input shape  : ";
    util::print_tensor_shape(input_tensor);
    std::cout << std::endl;
    std::cout << "[DEBUG] Input tensor type: " << input_tensor->type << std::endl;

    // Preprocess input data
    cv::Mat preprocessed_image = util::preprocess_image(origin_image, input_height, input_width);
    process_input_tensor(interpreter, preprocessed_image);

    util::timer_stop("Preprocessing");

    /* Warmup and Inference */
    run_warmup(interpreter, delegate);
    run_inference(interpreter, delegate, profiler, enable_profiling);

    /* PostProcessing */
    util::timer_start("Postprocessing");

    // Get output tensor
    TfLiteTensor *output_tensor = interpreter->output_tensor(0);
    std::cout << "[INFO] Output shape : ";
    util::print_tensor_shape(output_tensor);
    std::cout << std::endl;
    std::cout << "[DEBUG] Output tensor type: " << output_tensor->type << std::endl;

    int num_classes = output_tensor->dims->data[1];
    std::vector<float> probs(num_classes);
    process_output_tensor(interpreter, probs);

    util::timer_stop("Postprocessing");
    util::timer_stop("E2E Total(Pre+Inf+Post)");

    /* Print Results */
    // Load class label mapping
    auto label_map = util::load_class_labels(label_path);

    // Print Top-5 results
    std::cout << "\n[INFO] Top 5 predictions:" << std::endl;
    auto top_k_indices = util::get_topK_indices(probs, 5);
    for (int idx : top_k_indices)
    {
        std::string label = label_map.count(idx) ? label_map[idx] : "unknown";
        std::cout << "- Class " << idx << " (" << label << "): " << probs[idx] << std::endl;
    }

    /* Print Timers */
    util::print_all_timers();

    /* Print Profiling Results */
    print_profiling_results(profiler, interpreter, enable_profiling);

    std::cout << "========================" << std::endl;

    /* Deallocate delegate */
    if (delegate)
    {
        if (delegate_type == "xnnpack")
        {
            TfLiteXNNPackDelegateDelete(delegate);
        }
        else if (delegate_type == "gpu")
        {
            TfLiteGpuDelegateV2Delete(delegate);
        }
    }
    return 0;
}

// Function implementations
void apply_xnnpack_delegate(std::unique_ptr<tflite::Interpreter> &interpreter,
                            TfLiteDelegate *&delegate, bool &delegate_applied, int num_threads)
{
    TfLiteXNNPackDelegateOptions xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
    xnnpack_opts.num_threads = num_threads;
    delegate = TfLiteXNNPackDelegateCreate(&xnnpack_opts);

    if (interpreter->ModifyGraphWithDelegate(delegate) == kTfLiteOk)
    {
        delegate_applied = true;
        std::cout << "[INFO] XNNPACK delegate applied successfully (threads: " << num_threads << ")" << std::endl;
    }
    else
    {
        std::cerr << "[ERROR] Failed to apply XNNPACK delegate" << std::endl;
    }
}

void apply_gpu_delegate(std::unique_ptr<tflite::Interpreter> &interpreter,
                        TfLiteDelegate *&delegate, bool &delegate_applied)
{
    TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
    gpu_opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
    gpu_opts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    gpu_opts.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    gpu_opts.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;

    delegate = TfLiteGpuDelegateV2Create(&gpu_opts);

    if (interpreter->ModifyGraphWithDelegate(delegate) == kTfLiteOk)
    {
        delegate_applied = true;
        std::cout << "[INFO] GPU delegate applied successfully" << std::endl;
    }
    else
    {
        std::cerr << "[ERROR] Failed to apply GPU delegate" << std::endl;
    }
}

void process_input_tensor(std::unique_ptr<tflite::Interpreter> &interpreter,
                          const cv::Mat &preprocessed_image)
{
    TfLiteTensor *input_tensor = interpreter->input_tensor(0);

    if (input_tensor->type == kTfLiteFloat32)
    {
        std::cout << "[INFO] Processing FP32 input path" << std::endl;
        float *input_tensor_buffer = interpreter->typed_input_tensor<float>(0);
        std::memcpy(input_tensor_buffer, preprocessed_image.ptr<float>(),
                    preprocessed_image.total() * preprocessed_image.elemSize());
    }
    else if (input_tensor->type == kTfLiteUInt8)
    {
        std::cout << "[INFO] Processing UINT8 input path" << std::endl;
        // Get quantization parameters
        TfLiteQuantization quantization = input_tensor->quantization;
        float scale = quantization.params ? ((TfLiteAffineQuantization *)quantization.params)->scale->data[0] : 1.0f;
        int32_t zero_point = quantization.params ? ((TfLiteAffineQuantization *)quantization.params)->zero_point->data[0] : 0;

        std::cout << "[DEBUG] Quantization - Scale: " << scale << ", Zero point: " << zero_point << std::endl;

        // Convert float32 to quantized uint8
        uint8_t *input_tensor_buffer = interpreter->typed_input_tensor<uint8_t>(0);
        const float *float_data = preprocessed_image.ptr<float>();
        size_t total_elements = preprocessed_image.total() * preprocessed_image.channels();

        for (size_t i = 0; i < total_elements; ++i)
        {
            int32_t quantized_value = static_cast<int32_t>(std::round(float_data[i] / scale) + zero_point);
            quantized_value = std::max(0, std::min(255, quantized_value));
            input_tensor_buffer[i] = static_cast<uint8_t>(quantized_value);
        }
    }
    else if (input_tensor->type == kTfLiteInt8)
    {
        std::cout << "[INFO] Processing INT8 (signed) input path" << std::endl;
        // Get quantization parameters
        TfLiteQuantization quantization = input_tensor->quantization;
        float scale = quantization.params ? ((TfLiteAffineQuantization *)quantization.params)->scale->data[0] : 1.0f;
        int32_t zero_point = quantization.params ? ((TfLiteAffineQuantization *)quantization.params)->zero_point->data[0] : 0;

        std::cout << "[DEBUG] Quantization - Scale: " << scale << ", Zero point: " << zero_point << std::endl;

        // Convert float32 to quantized int8
        int8_t *input_tensor_buffer = interpreter->typed_input_tensor<int8_t>(0);
        const float *float_data = preprocessed_image.ptr<float>();
        size_t total_elements = preprocessed_image.total() * preprocessed_image.channels();

        for (size_t i = 0; i < total_elements; ++i)
        {
            int32_t quantized_value = static_cast<int32_t>(std::round(float_data[i] / scale) + zero_point);
            quantized_value = std::max(-128, std::min(127, quantized_value));
            input_tensor_buffer[i] = static_cast<int8_t>(quantized_value);
        }
    }
    else
    {
        std::cerr << "[ERROR] Unsupported input tensor type: " << input_tensor->type << std::endl;
        exit(1);
    }
}

void process_output_tensor(std::unique_ptr<tflite::Interpreter> &interpreter,
                           std::vector<float> &probs)
{
    TfLiteTensor *output_tensor = interpreter->output_tensor(0);
    int num_classes = output_tensor->dims->data[1];

    if (output_tensor->type == kTfLiteFloat32)
    {
        std::cout << "[INFO] Processing FP32 output path" << std::endl;
        float *logits = interpreter->typed_output_tensor<float>(0);
        util::softmax(logits, probs, num_classes);
    }
    else if (output_tensor->type == kTfLiteUInt8)
    {
        std::cout << "[INFO] Processing UINT8 output path" << std::endl;
        // Get quantization parameters
        TfLiteQuantization quantization = output_tensor->quantization;
        float scale = quantization.params ? ((TfLiteAffineQuantization *)quantization.params)->scale->data[0] : 1.0f;
        int32_t zero_point = quantization.params ? ((TfLiteAffineQuantization *)quantization.params)->zero_point->data[0] : 0;

        std::cout << "[DEBUG] Output quantization - Scale: " << scale << ", Zero point: " << zero_point << std::endl;

        // Dequantize uint8 to float32
        uint8_t *quantized_logits = interpreter->typed_output_tensor<uint8_t>(0);
        std::vector<float> float_logits(num_classes);

        for (int i = 0; i < num_classes; ++i)
        {
            float_logits[i] = scale * (static_cast<int32_t>(quantized_logits[i]) - zero_point);
        }

        util::softmax(float_logits.data(), probs, num_classes);
    }
    else if (output_tensor->type == kTfLiteInt8)
    {
        std::cout << "[INFO] Processing INT8 output path" << std::endl;
        // Get quantization parameters
        TfLiteQuantization quantization = output_tensor->quantization;
        float scale = quantization.params ? ((TfLiteAffineQuantization *)quantization.params)->scale->data[0] : 1.0f;
        int32_t zero_point = quantization.params ? ((TfLiteAffineQuantization *)quantization.params)->zero_point->data[0] : 0;

        std::cout << "[DEBUG] Output quantization - Scale: " << scale << ", Zero point: " << zero_point << std::endl;

        // Dequantize int8 to float32
        int8_t *quantized_logits = interpreter->typed_output_tensor<int8_t>(0);
        std::vector<float> float_logits(num_classes);

        for (int i = 0; i < num_classes; ++i)
        {
            float_logits[i] = scale * (static_cast<int32_t>(quantized_logits[i]) - zero_point);
        }

        util::softmax(float_logits.data(), probs, num_classes);
    }
    else
    {
        std::cerr << "[ERROR] Unsupported output tensor type: " << output_tensor->type << std::endl;
        exit(1);
    }
}

void run_warmup(std::unique_ptr<tflite::Interpreter> &interpreter, TfLiteDelegate *xnn_delegate)
{
    std::cout << "[INFO] Running " << WARMUP_RUNS << " warmup iterations..." << std::endl;
    for (int i = 0; i < WARMUP_RUNS; ++i)
    {
        if (interpreter->Invoke() != kTfLiteOk)
        {
            std::cerr << "Failed to invoke interpreter during warmup" << std::endl;
            if (xnn_delegate)
            {
                TfLiteXNNPackDelegateDelete(xnn_delegate);
            }
            exit(1);
        }
    }
    std::cout << "[INFO] Warmup completed" << std::endl;
}

void run_inference(std::unique_ptr<tflite::Interpreter> &interpreter, TfLiteDelegate *xnn_delegate,
                   std::unique_ptr<tflite::profiling::BufferedProfiler> &profiler, bool enable_profiling)
{
    // Start profiling after warmup
    if (enable_profiling && profiler)
    {
        profiler->StartProfiling();
        std::cout << "[INFO] Profiler started after warmup" << std::endl;
    }

    util::timer_start("Inference");

    if (enable_profiling && profiler)
    {
        // Multiple runs for profiling accuracy
        std::cout << "[INFO] Running " << PROFILING_RUNS << " profiling iterations..." << std::endl;
        for (int i = 0; i < PROFILING_RUNS; ++i)
        {
            if (interpreter->Invoke() != kTfLiteOk)
            {
                std::cerr << "Failed to invoke interpreter during profiling" << std::endl;
                if (xnn_delegate)
                {
                    TfLiteXNNPackDelegateDelete(xnn_delegate);
                }
                exit(1);
            }
        }
    }
    else
    {
        // Single run for normal execution
        if (interpreter->Invoke() != kTfLiteOk)
        {
            std::cerr << "Failed to invoke interpreter" << std::endl;
            if (xnn_delegate)
            {
                TfLiteXNNPackDelegateDelete(xnn_delegate);
            }
            exit(1);
        }
    }

    util::timer_stop("Inference");

    // Stop profiling if enabled
    if (enable_profiling && profiler)
    {
        profiler->StopProfiling();
        std::cout << "[INFO] Ran " << PROFILING_RUNS << " profiling iterations" << std::endl;
        std::cout << "[INFO] Profiler stopped" << std::endl;
    }
}

void print_profiling_results(std::unique_ptr<tflite::profiling::BufferedProfiler> &profiler,
                             std::unique_ptr<tflite::Interpreter> &interpreter, bool enable_profiling)
{
    if (enable_profiling && profiler)
    {
        std::cout << "\n[INFO] ===== Detailed Profiling Results =====" << std::endl;

        auto profile_events = profiler->GetProfileEvents();
        std::cout << "[INFO] Total profile events captured: " << profile_events.size() << std::endl;

        if (!profile_events.empty())
        {
            tflite::profiling::ProfileSummarizer summarizer;
            summarizer.ProcessProfiles(profile_events, *interpreter);
            std::string summary_output = summarizer.GetOutputString();

            std::cout << summary_output << std::endl;
        }
        else
        {
            std::cout << "[WARNING] No profiling events captured. This may happen when:" << std::endl;
            std::cout << "  - XNNPACK delegate is applied (operations are fused)" << std::endl;
            std::cout << "  - Model is too simple or executes too quickly" << std::endl;
            std::cout << "  - Profiler was not set up correctly" << std::endl;
        }
    }
}