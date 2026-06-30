"""
fix_custom_onnx.py — Strip post-processing from custom YOLO ONNX for TRT 8.2.
Removes Mod and GatherElements ops by exposing raw detection tensor.
"""

import onnx, onnx_graphsurgeon as gs, numpy as np, sys, os

path = sys.argv[1] if len(sys.argv) > 1 else "yolo26_robocup15k.onnx"
out = sys.argv[2] if len(sys.argv) > 2 else "yolo26_custom_raw.onnx"

graph = gs.import_onnx(onnx.load(path))
all_t = graph.tensors()

# Find the Transpose in model.23 (raw output before post-processing)
transpose = None
for t_name, t in all_t.items():
    if "/model.23/Transpose_output_0" in t_name:
        transpose = t
        break

if transpose is None:
    print("ERROR: Could not find model.23/Transpose output")
    # Try alternative: find the last conv output before post-processing
    for n in reversed(graph.nodes):
        if n.op == "Conv":
            print(f"Last Conv: {n.name} -> {[o.name for o in n.outputs]}")
            break
    sys.exit(1)

print(f"Found Transpose output: {transpose.name}, shape={transpose.shape}")

# Set dtype and shape
transpose.dtype = np.float32
transpose.shape = (1, 8400, 84)  # 84 channels for 11 classes: 4 bbox + 11 classes... 
# Actually this model might have different number of classes
# Let's figure out from the output

# Count total output values before post-processing
# The transpose goes through Split which tells us channel count
for n in graph.nodes:
    if n.op == "Split" and "model.23" in n.name:
        print(f"Split node: {n.name}, outputs: {[o.name for o in n.outputs]}")

# Actually let's just look at what feeds the Transpose
for n in graph.nodes:
    if n.name == "/model.23/Transpose":
        inp = n.inputs[0]
        print(f"Transpose input: {inp.name}, shape={inp.shape}")
        # Find what Concat produced this
        for n2 in graph.nodes:
            if n2.op == "Concat" and inp.name in [o.name for o in n2.outputs]:
                print(f"  Pre-Concat: {n2.name}")
                for inp2 in n2.inputs:
                    print(f"    Input to Concat: {inp2.name}, shape={inp2.shape}")
                # Total channels = sum of all concat input channels
                total_channels = sum(s[1] for s in [i.shape for i in n2.inputs] if s and len(s) > 1)
                print(f"  Total channels: {total_channels}")
                transpose.shape = (1, total_channels, 8400)
                break

# Remove post-processing nodes (downstream of transpose)
def collect_downstream(start):
    c = set(); q = list(start.outputs)
    while q:
        n = q.pop(0)
        if n.name in c: continue
        c.add(n.name)
        for ot in n.outputs:
            for consumer in ot.outputs: q.append(consumer)
    return c

to_remove = collect_downstream(transpose)
print(f"Removing {len(to_remove)} post-processing nodes")
graph.nodes = [n for n in graph.nodes if n.name not in to_remove]
graph.outputs = [transpose]
transpose.name = "raw_detections"

graph.cleanup().toposort()
onnx_model = gs.export_onnx(graph)
onnx.save(onnx_model, out)

print(f"Saved: {out}")
print(f"Nodes: {len(onnx_model.graph.node)}, OneHot: {len([n for n in onnx_model.graph.node if n.op_type == 'OneHot'])}")
print(f"Outputs: {[o.name for o in onnx_model.graph.output]}")
