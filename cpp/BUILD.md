"""
Goal Detector — C++ Build & Deploy Guide
Jetson Nano 2GB | JP4.6.1 | TensorRT 8.2 | CUDA 10.2
"""

# ─── Export Model ─────────────────────────────────────────────────
# On laptop (WSL) or Jetson Python 3.8:
#
#   from ultralytics import YOLO
#   model = YOLO("yolov8n_robocup15k.pt")
#   model.export(format="onnx", imgsz=640, opset=11, simplify=True, nms=False, batch=1)
#
# Then on Jetson trtexec (or use Python TRT API on 3.6):
#
#   /usr/src/tensorrt/bin/trtexec \
#     --onnx=yolov8n_robocup15k.onnx \
#     --saveEngine=yolov8n_robocup15k.engine \
#     --workspace=512 \
#     --fp16 \
#     --buildOnly
#
# NOTE: --fp16 flag does NOT speed up Maxwell GPU (no FP16 tensor cores),
# but reduces memory bandwidth by 2x (smaller model = faster memory transfers).
# For pure speed on Maxwell, use --fp32 instead (avoid dequantization overhead).
#
# If trtexec fails with plugin errors, use the Python TRT API:
#
#   python3 << 'PYEOF'
#   import tensorrt as trt, os
#   logger = trt.Logger(trt.Logger.WARNING)
#   builder = trt.Builder(logger)
#   network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#   parser = trt.OnnxParser(network, logger)
#   with open("yolov8n_robocup15k.onnx", "rb") as f:
#       parser.parse(f.read())
#   config = builder.create_builder_config()
#   config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 29)  # 512 MB
#   profile = builder.create_optimization_profile()
#   inp = network.get_input(0)
#   profile.set_shape(inp.name, (1,3,640,640), (1,3,640,640), (1,3,640,640))
#   config.add_optimization_profile(profile)
#   plan = builder.build_serialized_network(network, config)
#   with open("yolov8n_robocup15k.engine", "wb") as f:
#       f.write(plan)
#   print(f"Engine: {os.path.getsize('yolov8n_robocup15k.engine')/1024/1024:.1f} MB")
#   PYEOF

# ─── Compile on Jetson ─────────────────────────────────────────────
#
#   cd ~/goal-detector/cpp
#   mkdir -p build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j2
#
# Required packages:
#   sudo apt install -y cmake g++ libopencv-dev

# ─── Run ───────────────────────────────────────────────────────────
#
#   sudo nvpmodel -m 0 && sudo jetson_clocks
#   cd ~/goal-detector/cpp/build && ./goal-detector
#
# The app loads yolov8n_robocup15k.engine from ~/goal-detector/.
# Shows OpenCV window with detections. Press ESC to quit.

# ─── Configuration ─────────────────────────────────────────────────
# Edit config.json at ~/goal-detector/config/config.json:
#   - model_path: path to .engine file
#   - model_input_size: 640 (must match engine)
#   - confidence_threshold: 0.25
#   - nms_threshold: 0.45
#   - goal_polygon: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
#   - udp_target_ip: "192.168.123.255"
#   - udp_target_port: 5000
#   - udp_broadcast: true
#   - camera_id: 0
#   - camera_width: 640
#   - camera_height: 480

# ─── Performance Expectations ──────────────────────────────────────
# Estimate: 5-10 FPS (vs 1.5 FPS Python) on Jetson Nano 2GB.
# Gain comes from: no Python overhead, no PyTorch RAM thrashing,
# no garbage collection, zero-copy CUDA buffers in C++.
# Real 15-20 FPS requires Xavier NX or Orin hardware.
