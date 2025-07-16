#include "util.hpp"

void util::print_tensor_shape(const TfLiteTensor *tensor)
{
    printf("[");
    for (int i = 0; i < tensor->dims->size; ++i)
    {
        printf("%d", tensor->dims->data[i]);
        if (i < tensor->dims->size - 1)
            printf(", ");
    }
    printf("]");
}

void util::print_model_summary(tflite::Interpreter *interpreter, bool delegate_applied)
{
    printf("\n[INFO] Model Summary \n");
    printf("📥 Input tensor count  : %zu\n", interpreter->inputs().size());
    printf("📤 Output tensor count : %zu\n", interpreter->outputs().size());
    printf("📦 Total tensor count  : %ld\n", interpreter->tensors_size());
    printf("🔧 Node (op) count     : %zu\n", interpreter->nodes_size());
    printf("🧩 Delegate applied    : %s\n", delegate_applied ? "Yes ✅" : "No ❌");
}

// Get indices of top-k highest values
std::vector<int> util::get_topK_indices(const std::vector<float> &data, int k)
{
    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(), indices.begin() + k, indices.end(),
        [&data](int a, int b)
        { return data[a] > data[b]; });
    indices.resize(k);
    return indices;
}

// Load label file from JSON and return index → label map
std::unordered_map<int, std::string> util::load_class_labels(const std::string &json_path)
{
    std::ifstream ifs(json_path, std::ifstream::binary);
    if (!ifs.is_open())
        throw std::runtime_error("Failed to open label file: " + json_path);

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errs;

    if (!Json::parseFromStream(builder, ifs, &root, &errs))
        throw std::runtime_error("Failed to parse JSON: " + errs);

    std::unordered_map<int, std::string> label_map;

    for (const auto &key : root.getMemberNames())
    {
        int idx = std::stoi(key);
        // Explicitly use string key to avoid string_view overloads
        const Json::Value& keyValue = root[static_cast<const Json::String&>(key)];
        if (keyValue.isArray() && keyValue.size() >= 2)
        {
            label_map[idx] = keyValue[1].asString(); // label = second element
        }
    }

    return label_map;
}


void util::timer_start(const std::string &label)
{
    util::timer_map[label] = util::TimerResult{util::Clock::now(), util::TimePoint{}, util::global_index++};
}

void util::timer_stop(const std::string &label)
{
    auto it = util::timer_map.find(label);
    if (it != timer_map.end())
    {
        it->second.end = Clock::now();
        it->second.stop_index = global_index++;
    }
    else
    {
        std::cerr << "[WARN] No active timer for label: " << label << std::endl;
    }
}

void util::print_all_timers()
{
    std::vector<std::pair<std::string, util::TimerResult>> ordered(util::timer_map.begin(), util::timer_map.end());
    std::sort(ordered.begin(), ordered.end(),
              [](const auto &a, const auto &b)
              {
                  return a.second.stop_index < b.second.stop_index; // ascend
              });

    std::cout << "\n[INFO] Elapsed time summary" << std::endl;
    for (const auto &[label, record] : ordered)
    {
        if (record.end != util::TimePoint{})
        {
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(record.end - record.start).count();
            auto ms = us / 1000.0;
            std::cout << "- " << label << " took " << ms << " ms (" << us << " us)" << std::endl;
        }
    }
}


// Preprocess: load, resize, center crop, RGB → float32 + normalize
cv::Mat util::preprocess_image(cv::Mat &image, int target_height, int target_width)
{
    int h = image.rows, w = image.cols;
    float scale = 256.0f / std::min(h, w);
    int new_h = static_cast<int>(h * scale);
    int new_w = static_cast<int>(w * scale);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int x = (new_w - target_width) / 2;
    int y = (new_h - target_height) / 2;
    cv::Rect crop(x, y, target_width, target_height);

    cv::Mat cropped = resized(crop);
    cv::Mat rgb_image;
    cv::cvtColor(cropped, rgb_image, cv::COLOR_BGR2RGB);

    // Normalize to float32
    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};

    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);
    for (int c = 0; c < 3; ++c)
        channels[c] = (channels[c] - mean[c]) / std[c];
    cv::merge(channels, float_image);

    return float_image;
}

// Apply softmax to logits
void util::softmax(const float *logits, std::vector<float> &probs, int size)
{
    float max_val = *std::max_element(logits, logits + size);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    if (sum > 0.0f)
    {
        for (int i = 0; i < size; ++i)
        {
            probs[i] /= sum;
        }
    }
}

//*==========================================*/
