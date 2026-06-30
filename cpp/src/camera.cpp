#include "camera.hpp"
#include <chrono>

Camera::~Camera() { stop(); }

bool Camera::open(int cam_id, int width, int height, int fps) {
    if (!m_cap.open(cam_id, cv::CAP_V4L2)) {
        std::cerr << "[CAM] Failed to open camera " << cam_id << std::endl;
        return false;
    }

    m_cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    m_cap.set(cv::CAP_PROP_FPS, fps);

    // Set MJPEG format for lower USB bandwidth
    m_cap.set(cv::CAP_PROP_FOURCC,
              cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    std::cout << "[CAM] Opened camera " << cam_id << ": "
              << (int)m_cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << (int)m_cap.get(cv::CAP_PROP_FRAME_HEIGHT)
              << " @" << (int)m_cap.get(cv::CAP_PROP_FPS) << " FPS" << std::endl;
    return true;
}

void Camera::start() {
    if (m_running) return;
    m_running = true;
    m_thread = std::thread(&Camera::capture_loop, this);
}

void Camera::stop() {
    m_running = false;
    if (m_thread.joinable()) m_thread.join();
    if (m_cap.isOpened()) m_cap.release();
}

bool Camera::read(cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_latest.empty()) return false;
    m_latest.copyTo(frame);
    return true;
}

void Camera::capture_loop() {
    cv::Mat frame;
    while (m_running) {
        if (!m_cap.read(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Update FPS
        m_frame_count++;
        double now = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        if (now - m_fps_timer >= 1.0) {
            m_fps = m_frame_count / (now - m_fps_timer);
            m_frame_count = 0;
            m_fps_timer = now;
        }

        // Store latest frame
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            frame.copyTo(m_latest);
        }
    }
}
