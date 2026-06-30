"""
model_tools.py — Model preparation utilities

Handles YOLO26n → ONNX → TensorRT engine conversion.
Run this on a machine with the appropriate GPU/software stack.
"""

import os
import sys
import subprocess
import argparse


def export_pytorch_to_onnx(pt_path, output_path=None, input_size=640):
    """Export YOLO26n PyTorch model to ONNX.

    This should be run on the Jetson (or any machine with ultralytics installed).
    """
    from ultralytics import YOLO

    if output_path is None:
        base = os.path.splitext(os.path.basename(pt_path))[0]
        output_path = f"{base}.onnx"

    print(f"[EXPORT] Loading model: {pt_path}")
    model = YOLO(pt_path)

    print(f"[EXPORT] Exporting to ONNX: {output_path}")
    success = model.export(
        format="onnx",
        imgsz=input_size,
        opset=12,
        simplify=True,
    )

    if success:
        print(f"[EXPORT] ONNX exported: {success}")
        return success
    return output_path


def build_tensorrt_engine(onnx_path, output_path=None, precision="fp32",
                          workspace_gb=1, max_batch=1):
    """Build TensorRT engine from ONNX using trtexec CLI.

    For Jetson Nano: use fp32 or int8.
    fp16 has limited benefit on Maxwell GPU.

    Args:
        onnx_path: Path to input ONNX model
        output_path: Path for output .engine file
        precision: "fp32", "fp16", or "int8"
        workspace_gb: Max workspace in GB (Nano has 2GB total, use 1)
        max_batch: Maximum batch size
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(onnx_path))[0]
        output_path = f"{base}.engine"

    # Find trtexec
    trtexec_paths = [
        "/usr/src/tensorrt/bin/trtexec",
        os.path.expanduser("~/trtexec"),
        os.path.join(os.path.dirname(sys.executable), "trtexec"),
    ]
    trtexec = None
    for p in trtexec_paths:
        if os.path.exists(p):
            trtexec = p
            break
    if trtexec is None:
        trtexec = "trtexec"  # Try PATH

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        f"--workspace={workspace_gb * 1024}",
        f"--minShapes=input:1x3x{640}x{640}",
        f"--optShapes=input:1x3x{640}x{640}",
        f"--maxShapes=input:1x3x{640}x{640}",
        "--buildOnly",
        "--noBuilderCache",
    ]

    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.append("--int8")
        # INT8 requires calibration data
        cmd.append(f"--calib=/home/nano/calibration_images")

    print(f"[TRT] Building engine: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"[TRT] Engine built: {output_path}")
        return output_path
    else:
        print(f"[TRT] Build failed (exit {result.returncode})")
        print(result.stderr[-500:])
        return None


def build_engine_ultralytics(pt_path, output_path=None):
    """Use Ultralytics export API to build TensorRT engine.

    This runs the ultralytics export pipeline which handles ONNX→TRT internally.

    For Jetson Nano, recommended flags:
        - format="engine"
        - half=False (Maxwell has no native FP16)
        - int8=True for best performance (requires calibration)
    """
    from ultralytics import YOLO

    if output_path is None:
        base = os.path.splitext(os.path.basename(pt_path))[0]
        output_path = f"{base}.engine"

    print(f"[EXPORT] Loading model: {pt_path}")
    model = YOLO(pt_path)

    print(f"[EXPORT] Exporting to TensorRT: {output_path}")
    result = model.export(
        format="engine",
        imgsz=640,
        half=False,
        simplify=True,
        workspace=1,  # 1GB workspace limit for Nano
    )
    print(f"[EXPORT] Result: {result}")
    return result


def download_yolo26n(output_dir="."):
    """Download the official YOLO26n PyTorch model."""
    from ultralytics import YOLO

    print(f"[DOWNLOAD] Fetching yolo26n.pt...")
    model = YOLO("yolo26n.pt")  # Downloads automatically
    out_path = os.path.join(output_dir, "yolo26n.pt")
    os.makedirs(output_dir, exist_ok=True)
    model.export(format="torchscript")  # Force save
    print(f"[DOWNLOAD] Model available at ~/.cache/ultralytics/yolo26n.pt")
    return str(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Goal Detector Model Tools")
    parser.add_argument("action", choices=["download", "export_onnx", "build_trt",
                                           "export_engine"])
    parser.add_argument("--input", help="Input model path")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--workspace", type=int, default=1, help="Max workspace GB")

    args = parser.parse_args()

    if args.action == "download":
        download_yolo26n(".")
    elif args.action == "export_engine":
        build_engine_ultralytics(args.input, args.output)
    elif args.action == "export_onnx":
        export_pytorch_to_onnx(args.input, args.output)
    elif args.action == "build_trt":
        build_tensorrt_engine(args.input, args.output, args.precision, args.workspace)
