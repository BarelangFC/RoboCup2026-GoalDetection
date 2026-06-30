#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>

class Camera {
public:
    Camera() = default;
    ~Camera();

    bool open(int cam_id, int width, int height, int fps = 30);
    void start();
    void stop();

    // Get latest frame (thread-safe)
    bool read(cv::Mat& frame);

    double get_fps() const { return m_fps; }

private:
    cv::VideoCapture m_cap;
    std::thread m_thread;
    std::mutex m_mutex;
    cv::Mat m_latest;
    std::atomic<bool> m_running{false};
    double m_fps = 0.0;
    int m_frame_count = 0;
    double m_fps_timer = 0.0;

    void capture_loop();
};

#endif
