
import onnx
import onnx_graphsurgeon as gs
import numpy as np
import os

graph = gs.import_onnx(onnx.load("yolo26n.onnx"))
all_tensors = graph.tensors()

# Find the transpose output (raw [batch, 8400, 84] tensor)
transpose_out = all_tensors["/model.23/Transpose_output_0"]
print(f"Transpose consumers: {[c.name for c in transpose_out.outputs]}")

# Collect all nodes downstream from Transpose
def collect_downstream(start_tensor):
    collected = set()
    queue = list(start_tensor.outputs)
    while queue:
        node = queue.pop(0)
        if node.name in collected:
            continue
        collected.add(node.name)
        print(f"  Queue: {node.name} ({node.op})")
        for out_t in node.outputs:
            for consumer in out_t.outputs:
                queue.append(consumer)
    return collected

to_remove = collect_downstream(transpose_out)
print(f"\nNodes to remove: {len(to_remove)}")

# Remove them
graph.nodes = [n for n in graph.nodes if n.name not in to_remove]

# Set output to the transpose output
graph.outputs = [transpose_out]
transpose_out.name = "raw_detections"

# Cleanup
graph.cleanup().toposort()

# Save
out_path = "yolo26n_raw.onnx"
onnx_model = gs.export_onnx(graph)

# Fix output metadata
for out in onnx_model.graph.output:
    out.type.tensor_type.elem_type = 1  # FLOAT

onnx.save(onnx_model, out_path)

# Verify
model2 = onnx.load(out_path)
onehot_count = len([n for n in model2.graph.node if n.op_type == "OneHot"])
print(f"\nResult: {len(model2.graph.node)} nodes, {len(model2.graph.output)} outputs")
print(f"OneHot: {onehot_count}")
print(f"Outputs: {[o.name for o in model2.graph.output]}")
print(f"Size: {os.path.getsize(out_path)/1024/1024:.1f} MB")
