#include "yolo.hpp"
#include <cmath>
#include <algorithm>

void YoloProcessor::preprocess(const cv::Mat& frame, float* blob,
                                float& scale, float& pad_x, float& pad_y) {
    int h = frame.rows;
    int w = frame.cols;

    // Letterbox resize
    scale = std::min((float)cfg.input_w / w, (float)cfg.input_h / h);
    int nw = (int)(w * scale);
    int nh = (int)(h * scale);
    pad_x = (cfg.input_w - nw) / 2.0f;
    pad_y = (cfg.input_h - nh) / 2.0f;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

    // Create padded canvas
    cv::Mat canvas(cfg.input_h, cfg.input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect((int)pad_x, (int)pad_y, nw, nh)));

    // Convert to NCHW float [0,1]
    int idx = 0;
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < cfg.input_h; i++) {
            for (int j = 0; j < cfg.input_w; j++) {
                blob[idx++] = canvas.at<cv::Vec3b>(i, j)[c] / 255.0f;
            }
        }
    }
}

std::vector<Detection> YoloProcessor::postprocess(const float* raw_output,
    float scale, float pad_x, float pad_y,
    int orig_w, int orig_h) {

    std::vector<Detection> dets;
    int stride = cfg.num_predictions;  // 8400

    // Raw output is channel-major: [15, 8400]
    // Layout: 4 bbox channels + 11 class channels
    // For each prediction i:
    //   cx = raw[0*8400 + i]
    //   cy = raw[1*8400 + i]
    //   w  = raw[2*8400 + i]
    //   h  = raw[3*8400 + i]
    //   cls_k = raw[(4+k)*8400 + i]

    for (int i = 0; i < cfg.num_predictions; i++) {
        // Get class 0 (ball) score
        float cls_raw = raw_output[(4 + cfg.ball_class_id) * stride + i];
        // Apply sigmoid only if TRT fused it (values > 1 indicate logits)
        float conf = (cls_raw > 1.0f) ? 1.0f / (1.0f + expf(-cls_raw)) : cls_raw;

        if (conf < cfg.conf_threshold) continue;

        float cx = raw_output[0 * stride + i];
        float cy = raw_output[1 * stride + i];
        float w  = raw_output[2 * stride + i];
        float h  = raw_output[3 * stride + i];

        // Convert [cx,cy,w,h] in grid space → [x1,y1,x2,y2] in original space
        float x1 = (cx - w/2 - pad_x) / scale;
        float y1 = (cy - h/2 - pad_y) / scale;
        float x2 = (cx + w/2 - pad_x) / scale;
        float y2 = (cy + h/2 - pad_y) / scale;

        // Clamp
        x1 = std::max(0.0f, std::min((float)orig_w, x1));
        y1 = std::max(0.0f, std::min((float)orig_h, y1));
        x2 = std::max(0.0f, std::min((float)orig_w, x2));
        y2 = std::max(0.0f, std::min((float)orig_h, y2));

        dets.push_back({x1, y1, x2, y2, conf, cfg.ball_class_id});
    }

    return nms(dets);
}

std::vector<Detection> YoloProcessor::nms(std::vector<Detection>& dets) {
    if (dets.size() <= 1) return dets;

    // Sort by confidence descending
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<Detection> result;
    std::vector<bool> removed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); i++) {
        if (removed[i]) continue;
        result.push_back(dets[i]);

        for (size_t j = i + 1; j < dets.size(); j++) {
            if (removed[j]) continue;

            // IoU calculation
            float xi1 = std::max(dets[i].x1, dets[j].x1);
            float yi1 = std::max(dets[i].y1, dets[j].y1);
            float xi2 = std::min(dets[i].x2, dets[j].x2);
            float yi2 = std::min(dets[i].y2, dets[j].y2);

            float inter = std::max(0.0f, xi2 - xi1) * std::max(0.0f, yi2 - yi1);
            float area_i = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1);
            float area_j = (dets[j].x2 - dets[j].x1) * (dets[j].y2 - dets[j].y1);
            float union_area = area_i + area_j - inter;

            if (union_area > 0 && inter / union_area > cfg.nms_threshold) {
                removed[j] = true;
            }
        }
    }
    return result;
}

void YoloProcessor::draw(cv::Mat& frame, const std::vector<Detection>& dets) {
    for (const auto& d : dets) {
        cv::rectangle(frame,
                      cv::Point((int)d.x1, (int)d.y1),
                      cv::Point((int)d.x2, (int)d.y2),
                      cv::Scalar(0, 255, 0), 2);

        // Center point
        cv::circle(frame,
                   cv::Point((int)((d.x1+d.x2)/2), (int)((d.y1+d.y2)/2)),
                   3, cv::Scalar(0, 255, 255), -1);

        char label[64];
        snprintf(label, sizeof(label), "ball %.2f", d.confidence);
        cv::putText(frame, label,
                    cv::Point((int)d.x1, (int)d.y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 1);
    }
}
