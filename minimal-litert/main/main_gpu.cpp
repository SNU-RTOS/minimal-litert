// gpu-delegate-main
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>

#include <opencv2/opencv.hpp> //opencv

#include "tflite/delegates/xnnpack/xnnpack_delegate.h" //for xnnpack delegate
#include "tflite/delegates/gpu/delegate.h"             // for gpu delegate
#include "tflite/model_builder.h"
#include "tflite/interpreter_builder.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model.h"
#include "util.hpp"


int main(int argc, char *argv[])
{
    std::cout << "====== main_gpu ====" << std::endl;

    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path> <label_json_path>" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string label_path = argv[3];

    // Determine model type from filename
    bool is_int8_model = (model_path.find("int8") != std::string::npos);
    std::cout << "[INFO] Model type detected: " << (is_int8_model ? "INT8" : "FP32") << std::endl;

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

    /* Apply GPU Delegate */
    util::timer_start("Apply Delegate");
    TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
    gpu_opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;

    TfLiteDelegate *gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
    bool delegate_applied = false;
    if (interpreter->ModifyGraphWithDelegate(gpu_delegate) == kTfLiteOk)
    {
        delegate_applied = true;
    }
    else
    {
        std::cerr << "Failed to apply GPU delegate" << std::endl;
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
        throw std::runtime_error("Failed to load image: " + image_path);
    util::timer_stop("Load Input Image");

    /* Preprocessing */
    util::timer_start("E2E Total(Pre+Inf+Post)");
    util::timer_start("Preprocessing");
    // Get input tensor info
    TfLiteTensor *input_tensor = interpreter->input_tensor(0);
    int input_height = input_tensor->dims->data[1];
    int input_width = input_tensor->dims->data[2];
    int input_channels = input_tensor->dims->data[3];

    std::cout << "\n[INFO] Input shape  : ";
    util::print_tensor_shape(input_tensor);
    std::cout << std::endl;
    std::cout << "[DEBUG] Input tensor type: " << input_tensor->type << std::endl;

    // Preprocess input data based on tensor type
    cv::Mat preprocessed_image = util::preprocess_image(origin_image, input_height, input_width);
    
    if (input_tensor->type == kTfLiteFloat32) {
        std::cout << "[INFO] Processing FP32 input path" << std::endl;
        // Copy HWC float32 cv::Mat to TFLite input tensor
        float *input_tensor_buffer = interpreter->typed_input_tensor<float>(0);
        std::memcpy(input_tensor_buffer, preprocessed_image.ptr<float>(),
                    preprocessed_image.total() * preprocessed_image.elemSize());
    }
    else if (input_tensor->type == kTfLiteUInt8) {
        std::cout << "[INFO] Processing INT8 input path" << std::endl;
        // Get quantization parameters using TensorFlow Lite API
        TfLiteQuantization quantization = input_tensor->quantization;
        float scale = quantization.params ? 
            ((TfLiteAffineQuantization*)quantization.params)->scale->data[0] : 1.0f;
        int32_t zero_point = quantization.params ? 
            ((TfLiteAffineQuantization*)quantization.params)->zero_point->data[0] : 0;
        
        std::cout << "[DEBUG] Quantization - Scale: " << scale << ", Zero point: " << zero_point << std::endl;
        
        // Convert float32 to quantized uint8
        uint8_t *input_tensor_buffer = interpreter->typed_input_tensor<uint8_t>(0);
        float* float_data = preprocessed_image.ptr<float>();
        size_t total_elements = preprocessed_image.total() * preprocessed_image.channels();
        
        for (size_t i = 0; i < total_elements; ++i) {
            int32_t quantized_value = static_cast<int32_t>(std::round(float_data[i] / scale) + zero_point);
            quantized_value = std::max(0, std::min(255, quantized_value));
            input_tensor_buffer[i] = static_cast<uint8_t>(quantized_value);
        }
    }
    else if (input_tensor->type == kTfLiteInt8) {
        std::cout << "[INFO] Processing INT8 (signed) input path" << std::endl;
        // Get quantization parameters using TensorFlow Lite API
        TfLiteQuantization quantization = input_tensor->quantization;
        float scale = quantization.params ? 
            ((TfLiteAffineQuantization*)quantization.params)->scale->data[0] : 1.0f;
        int32_t zero_point = quantization.params ? 
            ((TfLiteAffineQuantization*)quantization.params)->zero_point->data[0] : 0;
        
        std::cout << "[DEBUG] Quantization - Scale: " << scale << ", Zero point: " << zero_point << std::endl;
        
        // Convert float32 to quantized int8
        int8_t *input_tensor_buffer = interpreter->typed_input_tensor<int8_t>(0);
        float* float_data = preprocessed_image.ptr<float>();
        size_t total_elements = preprocessed_image.total() * preprocessed_image.channels();
        
        for (size_t i = 0; i < total_elements; ++i) {
            int32_t quantized_value = static_cast<int32_t>(std::round(float_data[i] / scale) + zero_point);
            quantized_value = std::max(-128, std::min(127, quantized_value));
            input_tensor_buffer[i] = static_cast<int8_t>(quantized_value);
        }
    }
    else {
        std::cerr << "[ERROR] Unsupported input tensor type: " << input_tensor->type << std::endl;
        return 1;
    }

    util::timer_stop("Preprocessing");

    /* Inference */
    util::timer_start("Inference");

    if (interpreter->Invoke() != kTfLiteOk)
    {
        std::cerr << "Failed to invoke interpreter" << std::endl;
        return 1;
    }
    util::timer_stop("Inference");

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

    // Handle different output tensor types
    if (output_tensor->type == kTfLiteFloat32) {
        std::cout << "[INFO] Processing FP32 output path" << std::endl;
        float *logits = interpreter->typed_output_tensor<float>(0);
        util::softmax(logits, probs, num_classes);
    }
    else if (output_tensor->type == kTfLiteUInt8) {
        std::cout << "[INFO] Processing UINT8 output path" << std::endl;
        // Get quantization parameters using TensorFlow Lite API
        TfLiteQuantization quantization = output_tensor->quantization;
        float scale = quantization.params ? 
            ((TfLiteAffineQuantization*)quantization.params)->scale->data[0] : 1.0f;
        int32_t zero_point = quantization.params ? 
            ((TfLiteAffineQuantization*)quantization.params)->zero_point->data[0] : 0;
        
        std::cout << "[DEBUG] Output quantization - Scale: " << scale << ", Zero point: " << zero_point << std::endl;
        
        // Dequantize uint8 to float32
        uint8_t *quantized_logits = interpreter->typed_output_tensor<uint8_t>(0);
        std::vector<float> float_logits(num_classes);
        
        for (int i = 0; i < num_classes; ++i) {
            float_logits[i] = scale * (static_cast<int32_t>(quantized_logits[i]) - zero_point);
        }
        
        util::softmax(float_logits.data(), probs, num_classes);
    }
    else if (output_tensor->type == kTfLiteInt8) {
        std::cout << "[INFO] Processing INT8 output path" << std::endl;
        // Get quantization parameters using TensorFlow Lite API
        TfLiteQuantization quantization = output_tensor->quantization;
        float scale = quantization.params ? 
            ((TfLiteAffineQuantization*)quantization.params)->scale->data[0] : 1.0f;
        int32_t zero_point = quantization.params ? 
            ((TfLiteAffineQuantization*)quantization.params)->zero_point->data[0] : 0;
        
        std::cout << "[DEBUG] Output quantization - Scale: " << scale << ", Zero point: " << zero_point << std::endl;
        
        // Dequantize int8 to float32
        int8_t *quantized_logits = interpreter->typed_output_tensor<int8_t>(0);
        std::vector<float> float_logits(num_classes);
        
        for (int i = 0; i < num_classes; ++i) {
            float_logits[i] = scale * (static_cast<int32_t>(quantized_logits[i]) - zero_point);
        }
        
        util::softmax(float_logits.data(), probs, num_classes);
    }
    else {
        std::cerr << "[ERROR] Unsupported output tensor type: " << output_tensor->type << std::endl;
        return 1;
    }

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
    std::cout << "========================" << std::endl;

    /* Deallocate delegate */
    if (gpu_delegate)
    {
        TfLiteGpuDelegateV2Delete(gpu_delegate);
    }
    return 0;
}
