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

namespace {

    // Inference mode enum
    enum class InferenceMode
    {
        WARMUP,
        RUN
    };

    struct BenchmarkConfig
    {
        std::string model_path;
        std::string image_path;
        std::string label_path;
        std::string profiling_result_path;
        bool enable_profiling = false;
        int num_threads = 4;
        int num_warmup = 5; // Default warmup runs
        int num_run = 1;    // Default profiling runs
        std::string delegate_type = "xnnpack";
    };

    // Vector of formatter/summarizer pairs for profiling output
    struct ProfilerOutput
    {
        std::shared_ptr<tflite::profiling::ProfileSummaryFormatter> formatter;
        std::shared_ptr<tflite::profiling::ProfileSummarizer> init_summarizer;
        std::shared_ptr<tflite::profiling::ProfileSummarizer> run_summarizer;
        std::string output_type; // "log" or "csv"
        std::string output_path; // empty for log, file path for csv
    };

    // Functions
    BenchmarkConfig parse_arguments(int argc, char *argv[])
    {
        BenchmarkConfig config;
        if (argc < 6 || argc > 9)
        {
            std::cerr << "[ERROR] Usage: " << argv[0] << " <model_path> <image_path> <label_json_path> <num_thread> <delegate_type> [csv_file_path] [warmup_runs] [profiling_runs]" << std::endl;
            std::cerr << "[ERROR]   delegate_type: xnnpack or gpu" << std::endl;
            exit(1);
        }
        config.model_path = argv[1];
        config.image_path = argv[2];
        config.label_path = argv[3];
        config.num_threads = (argc >= 5) ? std::atoi(argv[4]) : 4;
        config.delegate_type = (argc >= 6) ? std::string(argv[5]) : "xnnpack";
        config.profiling_result_path = (argc >= 7) ? argv[6] : "";
        config.enable_profiling = !config.profiling_result_path.empty();
        // Override warmup and profiling runs if provided
        config.num_warmup = (argc >= 8) ? std::atoi(argv[7]) : config.num_warmup;
        config.num_run = (argc >= 9) ? std::atoi(argv[8]) : config.num_run;

        std::cout << "[INFO] Model path: " << config.model_path << std::endl;
        std::cout << "[INFO] Image path: " << config.image_path << std::endl;
        std::cout << "[INFO] Label path: " << config.label_path << std::endl;
        std::cout << "[INFO] Profiling result path: " << config.profiling_result_path << std::endl;
        std::cout << "[INFO] Profiling enabled: " << (config.enable_profiling ? "YES" : "NO (CSV path not provided)") << std::endl;
        std::cout << "[INFO] Number of threads: " << config.num_threads << std::endl;
        std::cout << "[INFO] Delegate type: " << config.delegate_type << std::endl;
        std::cout << "[INFO] Warmup runs: " << config.num_warmup << std::endl;
        std::cout << "[INFO] Profiling runs: " << config.num_run << std::endl;

        return config;
    }


    void apply_delegate(std::unique_ptr<tflite::Interpreter> &interpreter,
                        TfLiteDelegate *&delegate, bool &delegate_applied,
                        const std::string &delegate_type, int num_threads)
    {
        delegate_applied = false;
        if (delegate_type == "xnnpack")
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
                TfLiteXNNPackDelegateDelete(delegate);
                delegate = nullptr;
            }
        }
        else if (delegate_type == "gpu")
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
                TfLiteGpuDelegateV2Delete(delegate);
                delegate = nullptr;
            }
        }
        else if (delegate_type == "none" || delegate_type.empty())
        {
            std::cout << "[INFO] No delegate applied." << std::endl;
            delegate = nullptr;
            delegate_applied = true;
        }
        else
        {
            std::cerr << "[ERROR] Unknown delegate type: " << delegate_type << std::endl;
            delegate = nullptr;
            delegate_applied = false;
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

    void run_model(std::unique_ptr<tflite::Interpreter> &interpreter,
                TfLiteDelegate *delegate, const std::string &delegate_type,
                std::unique_ptr<tflite::profiling::BufferedProfiler> &profiler,
                bool enable_profiling, InferenceMode mode,
                std::vector<ProfilerOutput> &profiler_outputs)
    {
        // Use the cleanup_delegate function for delegate cleanup

        int iter = (mode == InferenceMode::WARMUP) ? config.num_warmup : config.num_run;
        std::string phase = (mode == InferenceMode::WARMUP) ? "warm-up" : "run and profile";

        std::cout << "\n[INFO] Running " << iter << " " << phase << " iterations..." << std::endl;

        
        for (int i = 0; i < iter; ++i)
        {
            if (enable_profiling && profiler && mode == InferenceMode::RUN)
            {
                profiler->Reset();
                profiler->StartProfiling();
            }

            std::string timer_phase_name="Inference_" + std::to_string(i);

            if (mode == InferenceMode::RUN)
            {
                util::timer_start(timer_phase_name);
            }

            if (interpreter->Invoke() != kTfLiteOk)
            {
                std::cerr << "[ERROR] Failed to invoke interpreter during " << phase << std::endl;
                cleanup_delegate(delegate, delegate_type);
                exit(1);
            }
            
            if (mode == InferenceMode::RUN)
            {
                util::timer_stop(timer_phase_name);
            }

            if (enable_profiling && profiler && mode == InferenceMode::RUN)
            {
                profiler->StopProfiling();
                for (auto &out : profiler_outputs)
                {
                    out.run_summarizer->ProcessProfiles(profiler->GetProfileEvents(), *interpreter);
                }
            }
        }

        
        std::cout << "[INFO] " << phase << " completed" << std::endl;
    }

    void cleanup_delegate(TfLiteDelegate *&delegate, const std::string &delegate_type)
    {
        if (!delegate)
            return;
        if (delegate_type == "xnnpack")
        {
            TfLiteXNNPackDelegateDelete(delegate);
        }
        else if (delegate_type == "gpu")
        {
            TfLiteGpuDelegateV2Delete(delegate);
        }
        else if (delegate_type == "none" || delegate_type.empty())
        {
            // No delegate to clean up
        }
        else
        {
            // Future extensibility: add more delegate types here
            std::cout << "[WARN] Unknown delegate type for cleanup: " << delegate_type;
        }
        delegate = nullptr;
    }
}


int main(int argc, char *argv[])
{
    std::cout << "\n====== main_cpu with profiling ====" << std::endl;

    /* 0. Parse Argument */
    // Parse command line arguments
    BenchmarkConfig config = parse_arguments(argc, argv);

    // Determine model type from filename
    bool is_int8_model = (config.model_path.find("int8") != std::string::npos);

    std::cout << "[INFO] Model type detected: " << (is_int8_model ? "INT8" : "FP32") << std::endl;
    std::cout << "[INFO] Initializing profiler..." << std::endl;

    //======================================================
    /* 1. Load model */
    util::timer_start("Load Model");
    std::cout << "[INFO] Loading model from: " << config.model_path << std::endl;
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(config.model_path.c_str());
    if (!model)
    {
        std::cerr << "[ERROR] Failed to load model" << std::endl;
        return 1;
    }
    util::timer_stop("Load Model");

    /* 2. Build interpreter */
    util::timer_start("Build Interpreter");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    util::timer_stop("Build Interpreter");

    util::print_model_signature(interpreter.get());

    // Setup Profiler
    constexpr int kProfilingBufferHeadrooms = 512;
    int total_nodes = util::count_total_nodes(interpreter.get());
    if (total_nodes > kProfilingBufferHeadrooms)
        total_nodes += kProfilingBufferHeadrooms;
    auto profiler = std::make_unique<tflite::profiling::BufferedProfiler>(total_nodes, true);
    interpreter->SetProfiler(profiler.get());

    // Initialize profiler outputs
    std::vector<ProfilerOutput> profiler_outputs;

    // Always add log output
    ProfilerOutput pf_out_default;
    pf_out_default.formatter = std::make_shared<tflite::profiling::ProfileSummaryDefaultFormatter>();
    pf_out_default.init_summarizer = std::make_shared<tflite::profiling::ProfileSummarizer>(pf_out_default.formatter);
    pf_out_default.run_summarizer = std::make_shared<tflite::profiling::ProfileSummarizer>(pf_out_default.formatter);
    pf_out_default.output_type = "log";
    pf_out_default.output_path = "";
    profiler_outputs.push_back(pf_out_default);

    // Add CSV output if requested
    if (!config.profiling_result_path.empty())
    {
        ProfilerOutput pf_out_csv;
        pf_out_csv.formatter = std::make_shared<tflite::profiling::ProfileSummaryCSVFormatter>();
        pf_out_csv.init_summarizer = std::make_shared<tflite::profiling::ProfileSummarizer>(pf_out_csv.formatter);
        pf_out_csv.run_summarizer = std::make_shared<tflite::profiling::ProfileSummarizer>(pf_out_csv.formatter);
        pf_out_csv.output_type = "csv";
        pf_out_csv.output_path = config.profiling_result_path;
        profiler_outputs.push_back(pf_out_csv);
    }

    /* 3. Apply delegate */
    // Start Initialization Profiling
    profiler->Reset();
    profiler->StartProfiling();

    util::timer_start("Apply Delegate");
    TfLiteDelegate *delegate = nullptr;
    bool delegate_applied = false;
    apply_delegate(interpreter, delegate, delegate_applied, config.delegate_type, config.num_threads);
    if (!delegate_applied && config.delegate_type != "none")
    {
        std::cerr << "[ERROR] Failed to apply delegate: " << config.delegate_type << std::endl;
    }
    util::timer_stop("Apply Delegate");

    /* [PROFILE] Setup Profilers */

    // Default formatter/summarizer for log output
    auto log_formatter = std::make_shared<tflite::profiling::ProfileSummaryDefaultFormatter>();
    auto init_log_summarizer = std::make_shared<tflite::profiling::ProfileSummarizer>(log_formatter);
    auto run_log_summarizer = std::make_shared<tflite::profiling::ProfileSummarizer>(log_formatter);

    // CSV formatter/summarizer for CSV output
    auto csv_formatter = std::make_shared<tflite::profiling::ProfileSummaryCSVFormatter>();
    auto init_csv_summarizer = std::make_shared<tflite::profiling::ProfileSummarizer>(csv_formatter);
    auto run_csv_summarizer = std::make_shared<tflite::profiling::ProfileSummarizer>(csv_formatter);

    /* 4. Allocate Tensor */
    util::timer_start("Allocate Tensor");
    if (!interpreter || interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "[ERROR] Failed to initialize interpreter" << std::endl;
        return 1;
    }
    util::timer_stop("Allocate Tensor");

    util::print_model_summary(interpreter.get(), delegate_applied);

    // Finish Init Profiling
    profiler->StopProfiling();
    for (auto &out : profiler_outputs)
    {
        out.init_summarizer->ProcessProfiles(profiler->GetProfileEvents(), *interpreter);
    }
    //======================================================

    /* 5. Load input image */
    util::timer_start("Load Input Image");
    cv::Mat origin_image = cv::imread(config.image_path);
    if (origin_image.empty())
    {
        throw std::runtime_error("Failed to load image: " + config.image_path);
    }
    util::timer_stop("Load Input Image");

    /* 6. Preprocessing */
    util::timer_start("Preprocessing");

    init_log_summarizer->ProcessProfiles(profiler->GetProfileEvents(), *interpreter);
    TfLiteTensor *input_tensor = interpreter->input_tensor(0);
    util::print_tensor_shape(input_tensor, "input_tensor");

    // Preprocess input data
    int input_height = input_tensor->dims->data[1];
    int input_width = input_tensor->dims->data[2];
    cv::Mat preprocessed_image = util::preprocess_image(origin_image, input_height, input_width);
    process_input_tensor(interpreter, preprocessed_image);

    util::timer_stop("Preprocessing");

    /* 7. Warmup and Inference */
    // Run warmup and profiling for all outputs

    run_model(interpreter, delegate, config.delegate_type, profiler, config.enable_profiling,
              InferenceMode::WARMUP, profiler_outputs);
    run_model(interpreter, delegate, config.delegate_type, profiler, config.enable_profiling,
              InferenceMode::RUN, profiler_outputs);

    /* 8. PostProcessing */

    util::timer_start("Postprocessing");
    // Get output tensor
    TfLiteTensor *output_tensor = interpreter->output_tensor(0);
    util::print_tensor_shape(output_tensor, "output_tensor");

    // Process output tensor to get probabilities
    int num_classes = output_tensor->dims->data[1];
    std::vector<float> probs(num_classes);
    process_output_tensor(interpreter, probs);

    util::timer_stop("Postprocessing");

    /* 9. Print Results */
    // Load class label mapping and print Top-5 results
    util::print_topk_results(probs, util::load_class_labels(config.label_path));

    // Print all timers
    util::print_all_timers();

    // Print Ops-level profiling time (log)
    // Print all profiler outputs
    std::cout << "\n[INFO] Generating Ops-level profiling (log)\n"
              << std::endl;
    for (auto &out : profiler_outputs)
    {
        out.formatter->HandleOutput(out.init_summarizer->GetOutputString(),
                                    out.run_summarizer->GetOutputString(), out.output_path);
    }

    /* 10. Deallocate delegate */
    cleanup_delegate(delegate, config.delegate_type);
    return 0;
}
