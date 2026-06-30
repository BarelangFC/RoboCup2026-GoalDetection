#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <csignal>
#include <algorithm>
#include <unistd.h>
#include <opencv2/opencv.hpp>

#include "trt_infer.hpp"
#include "yolo.hpp"
#include "geometry.hpp"
#include "network.hpp"
#include "camera.hpp"
#include "cuda_utils.hpp"
#include "goal_sender.hpp"

// ─── Config ──────────────────────────────────────────────────────
struct Config {
    std::string model_path = "/home/nano/yolov8n_robocup15k.engine";
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int camera_id = 0;
    int cam_w = 640;
    int cam_h = 480;
    std::string udp_ip = "192.168.123.255";
    int udp_port = 5000;
    bool udp_broadcast = true;

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        std::string line;
        while (std::getline(f, line)) {
            auto colon = line.find(':');
            if (colon == std::string::npos) continue;
            auto q2 = line.find('"', colon);
            if (q2 == std::string::npos) continue;
            std::string val = line.substr(q2 + 1);
            val.erase(0, val.find_first_not_of(" \t\""));
            auto end = val.find_last_not_of(" \t,\"\r\n");
            if (end != std::string::npos) val = val.substr(0, end + 1);
            if (line.find("model_path") != std::string::npos && line.find("model_input") == std::string::npos)
                if (!val.empty()) model_path = val;
            if (line.find("udp_target_ip") != std::string::npos && !val.empty()) udp_ip = val;
            if (line.find("udp_target_port") != std::string::npos) udp_port = std::stoi(val);
            if (line.find("udp_broadcast") != std::string::npos) udp_broadcast = val == "true";
        }
        f.close();
        std::cout << "[CFG] Model: " << model_path << std::endl;
        return true;
    }
};

// ─── Button ──────────────────────────────────────────────────────
struct Button {
    cv::Rect rect;
    std::string label;
    cv::Scalar color;
    cv::Scalar hover_color;

    Button() {}
    Button(int x, int y, int w, int h, const std::string& lbl,
           cv::Scalar c = cv::Scalar(60, 60, 60))
        : rect(x, y, w, h), label(lbl), color(c),
          hover_color(c + cv::Scalar(30, 30, 30)) {}

    bool contains(int px, int py) const {
        return px >= rect.x && px < rect.x + rect.width &&
               py >= rect.y && py < rect.y + rect.height;
    }

    void draw(cv::Mat& frame, bool hovered = false) const {
        cv::Scalar c = hovered ? hover_color : color;
        cv::rectangle(frame, rect, c, -1);
        cv::rectangle(frame, rect, cv::Scalar(100, 100, 100), 2);
        int fs = std::min(rect.height / 2, 28);
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fs / 20.0, 2, &baseline);
        cv::putText(frame, label,
            cv::Point(rect.x + (rect.width - ts.width) / 2,
                      rect.y + (rect.height + ts.height) / 2),
            cv::FONT_HERSHEY_SIMPLEX, fs / 20.0, cv::Scalar(220, 220, 220), 2);
    }
};

// ─── App State ──────────────────────────────────────────────────
enum AppState { STATE_MENU, STATE_TEST, STATE_RUN, STATE_CALIBRATE };

struct App {
    AppState state = STATE_MENU;
    bool polygon_ready = false;
    std::vector<cv::Point> calib_points;
    std::vector<cv::Point> polygon;
    int team_id = 1;
    double goal_flash_until = 0;
    double flash_duration = 1.0;
    int hovered_btn = -1;
    std::vector<cv::Mat> frame_buffer;
    int frame_buffer_max = 5;
    int goal_seq = 0;
    bool goal_sent = false;  // only send once per goal event
};

static App* g_app = nullptr;
static std::vector<Button> g_menu_buttons;
static Button g_back_btn;

// ─── Mouse callback ────────────────────────────────────────────
static void on_mouse(int event, int x, int y, int, void*) {
    if (!g_app) return;

    if (event == cv::EVENT_MOUSEMOVE) {
        g_app->hovered_btn = -1;
        if (g_app->state == STATE_MENU) {
            for (size_t i = 0; i < g_menu_buttons.size(); i++)
                if (g_menu_buttons[i].contains(x, y)) g_app->hovered_btn = i;
        }
        return;
    }

    if (event != cv::EVENT_LBUTTONDOWN) return;

    // Menu buttons
    if (g_app->state == STATE_MENU) {
        for (size_t i = 0; i < g_menu_buttons.size(); i++) {
            if (!g_menu_buttons[i].contains(x, y)) continue;
            switch (i) {
                case 0: g_app->state = STATE_TEST; break;
                case 1: g_app->state = STATE_RUN; g_app->goal_seq = 0; break;
                case 2: g_app->state = STATE_CALIBRATE; g_app->calib_points.clear(); break;
                case 3: g_app->team_id = (g_app->team_id == 1) ? 2 : 1; break;
                case 4: exit(0); break;
            }
            return;
        }
    }

    // BACK button
    if (g_app->state != STATE_MENU) {
        if (g_back_btn.contains(x, y)) {
            g_app->state = STATE_MENU;
            return;
        }
        // Calibration taps
        if (g_app->state == STATE_CALIBRATE && g_app->calib_points.size() < 4) {
            g_app->calib_points.push_back(cv::Point(x, y));
            std::cout << "[CALIB] Point " << g_app->calib_points.size() << "/4: ("
                      << x << "," << y << ")" << std::endl;
            if (g_app->calib_points.size() >= 4) {
                g_app->polygon = g_app->calib_points;
                g_app->polygon_ready = true;
                std::cout << "[CALIB] Goal polygon saved" << std::endl;
                g_app->state = STATE_MENU;
            }
        }
    }
}

// ─── Signal handler ────────────────────────────────────────────
static volatile bool g_running = true;
void signal_handler(int) { g_running = false; }

// ─── Draw Menu ───────────────────────────────────────────────────
void draw_menu(cv::Mat& frame) {
    frame = cv::Scalar(30, 30, 30);  // dark background
    int fw = frame.cols, fh = frame.rows;

    // Title
    std::string title = "GOAL DETECTOR";
    cv::Size ts = cv::getTextSize(title, cv::FONT_HERSHEY_SIMPLEX, 1.2, 3, nullptr);
    cv::putText(frame, title, cv::Point((fw - ts.width) / 2, 60),
                cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(220, 220, 220), 3);

    std::string sub = "Select a mode";
    ts = cv::getTextSize(sub, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
    cv::putText(frame, sub, cv::Point((fw - ts.width) / 2, 95),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(150, 150, 150), 1);

    // Status
    std::vector<std::string> status = {
        "Model: loaded",
        "Cam: ready",
        g_app->polygon_ready ? "Goal: calibrated" : "Goal: not set"
    };
    for (size_t i = 0; i < status.size(); i++) {
        cv::Scalar c = (status[i].find("not") == std::string::npos)
            ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 200);
        cv::putText(frame, status[i], cv::Point(20, fh - 80 + i * 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, c, 1);
    }

    // Buttons
    int bw = 220, bh = 45;
    int start_y = 130;
    int gap = 12;
    std::vector<std::string> labels = {"TEST MODE", "RUN MODE", "CALIBRATE",
        std::string("TEAM: ") + (g_app->team_id == 1 ? "1" : "2"),
        "QUIT"};
    std::vector<cv::Scalar> colors = {
        cv::Scalar(20, 80, 20), cv::Scalar(20, 20, 80),
        cv::Scalar(20, 60, 80),
        g_app->team_id == 1 ? cv::Scalar(60, 40, 20) : cv::Scalar(20, 40, 60),
        cv::Scalar(60, 20, 20)
    };
    g_menu_buttons.clear();
    for (size_t i = 0; i < labels.size(); i++) {
        int bx = (fw - bw) / 2;
        int by = start_y + i * (bh + gap);
        Button btn(bx, by, bw, bh, labels[i], colors[i]);
        btn.draw(frame, g_app->hovered_btn == (int)i);
        g_menu_buttons.push_back(btn);
    }
}

// ─── Draw BACK button ─────────────────────────────────────────
void draw_back_button(cv::Mat& frame) {
    int fw = frame.cols;
    g_back_btn = Button(fw - 130, 10, 120, 45, "BACK", cv::Scalar(30, 30, 80));
    g_back_btn.draw(frame);
}

// ─── Draw overlay ─────────────────────────────────────────────
void draw_overlay(cv::Mat& frame, double cam_fps, double det_fps,
                  const std::vector<Detection>& dets, double now) {
    int fw = frame.cols, fh = frame.rows;
    char buf[128];

    // Top-left: FPS
    snprintf(buf, sizeof(buf), "Cam: %.1f FPS  Det: %.1f FPS", cam_fps, det_fps);
    cv::putText(frame, buf, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

    // Detection count
    snprintf(buf, sizeof(buf), "Balls: %zu", dets.size());
    cv::putText(frame, buf, cv::Point(10, 45), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

    // Team assignment (bottom-left)
    snprintf(buf, sizeof(buf), "Goal Team: %d", g_app->team_id);
    cv::putText(frame, buf, cv::Point(10, fh - 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

    // Goal flash — show in both TEST and RUN
    if ((g_app->state == STATE_TEST || g_app->state == STATE_RUN) && now < g_app->goal_flash_until) {
        cv::putText(frame, "GOAL!", cv::Point(fw / 2 - 100, fh / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(0, 0, 255), 6);
    }

    // Calibration instructions
    if (g_app->state == STATE_CALIBRATE) {
        snprintf(buf, sizeof(buf), "Tap corner %zu/4 of goal area",
                 g_app->calib_points.size() + 1);
        cv::putText(frame, buf, cv::Point(fw / 2 - 180, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 165, 0), 2);
        for (auto& p : g_app->calib_points)
            cv::circle(frame, p, 6, cv::Scalar(0, 255, 255), -1);
        if (g_app->calib_points.size() >= 2) {
            for (size_t i = 1; i < g_app->calib_points.size(); i++)
                cv::line(frame, g_app->calib_points[i-1], g_app->calib_points[i], cv::Scalar(0, 255, 255), 2);
        }
        if (g_app->calib_points.size() == 4)
            cv::line(frame, g_app->calib_points[3], g_app->calib_points[0], cv::Scalar(0, 255, 255), 2);
    }

    // Goal polygon
    if (g_app->polygon_ready && g_app->polygon.size() >= 3)
        cv::polylines(frame, g_app->polygon, true, cv::Scalar(0, 255, 0), 2);

    // Mode label top-right
    const char* mode_str = "";
    cv::Scalar mode_color(0, 255, 0);
    switch (g_app->state) {
        case STATE_TEST: mode_str = "TEST"; mode_color = cv::Scalar(0, 255, 0); break;
        case STATE_RUN:  mode_str = "RUN";  mode_color = cv::Scalar(0, 0, 255); break;
        case STATE_CALIBRATE: mode_str = "CALIB"; mode_color = cv::Scalar(255, 165, 0); break;
        default: break;
    }
    if (g_app->state != STATE_MENU) {
        int tw = cv::getTextSize(mode_str, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr).width;
        cv::putText(frame, mode_str, cv::Point(fw - tw - 10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2);
    }
}

// ─── Main ────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);

    std::string config_path = argc > 1 ? argv[1]
        : "/home/nano/goal-detector/config/config.json";

    Config cfg;
    cfg.load(config_path);

    // Camera
    Camera cam;
    if (!cam.open(cfg.camera_id, cfg.cam_w, cfg.cam_h, 30)) return 1;
    cam.start();

    // TRT engine
    TrtEngine engine;
    if (!engine.load(cfg.model_path)) return 1;

    // YOLO
    YoloProcessor yolo;
    yolo.cfg.conf_threshold = cfg.conf_threshold;
    yolo.cfg.nms_threshold = cfg.nms_threshold;
    yolo.cfg.num_predictions = engine.get_output_size(0) / 15;
    std::cout << "[YOLO] Preds: " << yolo.cfg.num_predictions
              << "  thr: " << yolo.cfg.conf_threshold << std::endl;

    // Goal checker
    GoalChecker goal_checker;

    // UDP
    UdpDispatcher udp;
    udp.setup(cfg.udp_ip, cfg.udp_port, true);

    // Goal sender (TCP to GameController)
    GoalSender gs;
    std::string gc_host = cfg.udp_ip;
    if (gc_host.find("255") != std::string::npos) gc_host = "192.168.123.58";
    int gc_port = 3737;  // GC port for goal frames
    gs.setup(gc_host, gc_port);
    std::cout << "[SEND] GC TCP target: " << gc_host << ":" << gc_port << std::endl;

    // App state
    App app;
    app.state = STATE_MENU;
    g_app = &app;

    // Buffers
    std::vector<float> input_blob(engine.get_input_size());
    std::vector<float> output_blob(engine.get_output_size(0));
    std::vector<float*> output_ptrs = {output_blob.data()};

    // Window
    int win_w = 800, win_h = 540;
    cv::namedWindow("Goal Detector", cv::WINDOW_NORMAL);
    cv::resizeWindow("Goal Detector", win_w, win_h);
    cv::setMouseCallback("Goal Detector", on_mouse);
    cv::waitKey(1); // ensure window created

    std::cout << "[MAIN] State: " << app.state << " (0=MENU)" << std::endl;

    // Performance
    double fps_timer = 0.0;
    int frame_count = 0;
    double det_fps = 0.0;

    cv::Mat frame, display(win_h, win_w, CV_8UC3);
    std::cout << "[MAIN] Entering main loop. State=" << app.state << std::endl;

    // ─── Main Loop ───────────────────────────────────────────
    while (g_running) {
        double now = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        // ─── MENU state ──────────────────────────────────────
        if (app.state == STATE_MENU) {
            draw_menu(display);
            cv::imshow("Goal Detector", display);
            int key = cv::waitKey(100);  // Processes GUI events
            if (key == 27 || key == 'q') g_running = false;
            // Menu state changes via mouse callback (on_mouse)
            continue;
        }

        // ─── Active mode: get frame ──────────────────────────
        if (!cam.read(frame)) {
            cv::waitKey(1);
            continue;
        }
        display = frame.clone();

        // ─── Inference ───────────────────────────────────────
        float scale, pad_x, pad_y;
        yolo.preprocess(display, input_blob.data(), scale, pad_x, pad_y);
        engine.infer(input_blob.data(), 1, output_ptrs);
        auto dets = yolo.postprocess(output_blob.data(), scale, pad_x, pad_y,
                                      display.cols, display.rows);

        // ─── Draw detections ─────────────────────────────────
        for (auto& d : dets) {
            cv::rectangle(display,
                cv::Point((int)d.x1, (int)d.y1),
                cv::Point((int)d.x2, (int)d.y2),
                cv::Scalar(0, 255, 0), 2);
            cv::circle(display,
                cv::Point((int)((d.x1+d.x2)/2), (int)((d.y1+d.y2)/2)),
                3, cv::Scalar(0, 255, 255), -1);
            char label[64];
            snprintf(label, sizeof(label), "ball %.2f", d.confidence);
            cv::putText(display, label, cv::Point((int)d.x1, (int)d.y1 - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
        }

        // ─── Frame buffer ─────────────────────────────────
        // Always push latest frame into ring buffer
        if (!frame.empty()) {
            app.frame_buffer.push_back(frame.clone());
            while ((int)app.frame_buffer.size() > app.frame_buffer_max)
                app.frame_buffer.erase(app.frame_buffer.begin());
        }

        // ─── Goal polygon ─────────────────────────────────
        if (app.polygon_ready && app.polygon.size() >= 3) {
            cv::polylines(display, app.polygon, true, cv::Scalar(0, 255, 0), 2);

            // Goal check in TEST and RUN
            if (app.state == STATE_TEST || app.state == STATE_RUN) {
                goal_checker.set_polygon(TEAM_1, app.polygon);
                goal_checker.set_mode(GOAL_FULL);
                bool ball_in_goal = false;
                float gx, gy;
                for (auto& d : dets) {
                    TeamId scorer = goal_checker.check_goals(d, gx, gy);
                    if (scorer != TEAM_NONE) {
                        ball_in_goal = true;
                        // Only trigger once per goal event
                        if (!app.goal_sent) {
                            app.goal_sent = true;
                            app.goal_flash_until = now + app.flash_duration;
                            app.goal_seq++;
                            std::cout << "[GOAL] #" << app.goal_seq
                                      << " sending " << app.frame_buffer.size() << " frames" << std::endl;

                            if (app.state == STATE_RUN && app.frame_buffer.size() >= 3) {
                                std::vector<std::vector<unsigned char>> jpegs;
                                for (auto& f : app.frame_buffer) {
                                    std::vector<unsigned char> jpg;
                                    cv::imencode(".jpg", f, jpg, {cv::IMWRITE_JPEG_QUALITY, 70});
                                    jpegs.push_back(jpg);
                                }
                                gs.send(app.team_id, app.goal_seq, jpegs);
                            }
                        }
                    }
                }
                // Reset when ball leaves the polygon
                if (!ball_in_goal) app.goal_sent = false;
            }
        }

        // ─── FPS ─────────────────────────────────────────────
        frame_count++;
        if (now - fps_timer >= 1.0) {
            det_fps = frame_count / (now - fps_timer);
            frame_count = 0;
            fps_timer = now;
        }

        // ─── Overlay + BACK ──────────────────────────────────
        draw_overlay(display, cam.get_fps(), det_fps, dets, now);
        draw_back_button(display);

        // ─── Show ────────────────────────────────────────────
        cv::imshow("Goal Detector", display);
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') g_running = false;
        // BACK button click handled by on_mouse callback
    }

    // ─── Shutdown ─────────────────────────────────────────────
    std::cout << "[MAIN] Shutdown..." << std::endl;
    g_app = nullptr;
    cam.stop();
    engine.unload();
    udp.close();
    cv::destroyAllWindows();
    std::cout << "[MAIN] Done." << std::endl;
    return 0;
}
