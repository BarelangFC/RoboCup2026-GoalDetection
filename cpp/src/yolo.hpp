#ifndef YOLO_H
#define YOLO_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

struct YoloConfig {
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int input_w = 640;
    int input_h = 640;
    int num_classes = 11;
    int ball_class_id = 0;
    int num_predictions = 8400;
};

class YoloProcessor {
public:
    YoloConfig cfg;

    // Preprocess: letterbox + normalize → flat float buffer
    // Returns scale factor and padding for later de-warp
    void preprocess(const cv::Mat& frame, float* blob, float& scale, float& pad_x, float& pad_y);

    // Postprocess: parse raw TRT output [15 * 8400] → detections
    // Output is channel-major: 4 bbox + 11 cls, 8400 predictions
    std::vector<Detection> postprocess(const float* raw_output,
        float scale, float pad_x, float pad_y,
        int orig_w, int orig_h);

    // NMS
    std::vector<Detection> nms(std::vector<Detection>& dets);

    // Draw detections on frame
    void draw(cv::Mat& frame, const std::vector<Detection>& dets);

private:
    inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
};

#endif
