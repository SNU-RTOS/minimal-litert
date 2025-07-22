
#include "util.hpp"

void util::print_model_signature(tflite::Interpreter *interpreter)
{
    std::cout << "[INFO] Model signature keys:";
    if (interpreter)
    {
        const std::vector<const std::string *> &keys = interpreter->signature_keys();
        std::cout << "The Model contains " << keys.size() << " signature key(s).";
        if (!keys.empty())
        {
            std::cout << "They are listed below: ";
            for (const std::string *key : keys)
            {
                std::cout << "-> Signature Key: " << *key;
            }
        }
    }
    else
    {
        std::cout << "The Model does not contain any signature keys.";
    }
    std::cout << std::endl;
}

void util::print_tensor_shape(const TfLiteTensor *tensor, const std::string &label)
{

    std::cout << "\n[INFO] Shape of " << label << ": ";
    std::cout << "[";
    for (int i = 0; i < tensor->dims->size; ++i)
    {
        std::cout << tensor->dims->data[i];
        if (i < tensor->dims->size - 1)
            std::cout << ", ";
    }
    std::cout << "]";
    std::cout << "\n";
}

void util::print_model_summary(tflite::Interpreter *interpreter, bool delegate_applied)
{
    std::cout << "\n[INFO] Model Summary " << std::endl;
    std::cout << "📥 Input tensor count  : " << interpreter->inputs().size() << std::endl;
    std::cout << "📤 Output tensor count : " << interpreter->outputs().size() << std::endl;
    std::cout << "📦 Total tensor count  : " << interpreter->tensors_size() << std::endl;
    std::cout << "🔧 Node (op) count     : " << interpreter->nodes_size() << std::endl;
    std::cout << "🧩 Delegate applied    : " << (delegate_applied ? "Yes ✅" : "No ❌") << std::endl;
}

int util::count_total_nodes(tflite::Interpreter *interpreter)
{
    int total_nodes = 0;
    if (!interpreter)
        return 0;
    for (int i = 0; i < interpreter->subgraphs_size(); ++i)
    {
        total_nodes += static_cast<int>(interpreter->subgraph(i)->nodes_size());
    }
    return total_nodes;
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
        const Json::Value &keyValue = root[static_cast<const Json::String &>(key)];
        if (keyValue.isArray() && keyValue.size() >= 2)
        {
            label_map[idx] = keyValue[1].asString(); // label = second element
        }
    }

    return label_map;
}

void util::print_topk_results(const std::vector<float> &probs, const std::unordered_map<int, std::string> &label_map)
{
    std::cout << "\n[INFO] Top 5 predictions:" << std::endl;
    auto top_k_indices = util::get_topK_indices(probs, 5);
    for (int idx : top_k_indices)
    {
        std::string label = label_map.count(idx) ? label_map.at(idx) : "unknown";
        std::cout << "- Class " << idx << " (" << label << "): " << probs[idx] << std::endl;
    }
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
        std::cout << "[WARN] No active timer for label: " << label << std::endl;
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
    std::cout << "\n";
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
