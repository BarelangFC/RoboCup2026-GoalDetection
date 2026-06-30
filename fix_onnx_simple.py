
import onnx
import onnx_graphsurgeon as gs
import numpy as np

# Load
graph = gs.import_onnx(onnx.load("yolo26n.onnx"))

# The output of /model.23/Split_1 gives us [bboxes (8400,4), class_scores (8400,80)]
# We want to make this the model output instead of going through OneHot post-processing

# Find the key tensors
split_1_out = [t for t in graph.tensors().values() if t.name == "/model.23/Split_1_output_0"][0]
split_1_out_1 = [t for t in graph.tensors().values() if t.name == "/model.23/Split_1_output_1"][0]

# Also find the raw pre-transpose output (Concat_3)
concat_3_out = [t for t in graph.tensors().values() if t.name == "/model.23/Concat_3_output_0"][0]
transpose = [t for t in graph.tensors().values() if t.name == "/model.23/Transpose_output_0"][0]

print(f"Split_1_output_0 shape: {split_1_out.shape}, dtype: {split_1_out.dtype}")
print(f"Split_1_output_1 shape: {split_1_out_1.shape}, dtype: {split_1_out_1.dtype}")
print(f"Transpose_output_0 shape: {transpose.shape}, dtype: {transpose.dtype}")

# Remove all nodes that are ONLY used for post-processing (after Split_1)
# Strategy: find the subgraph from Split_1 to output0 and remove it
# Then make Split_1 outputs the new graph outputs

# Get all tensor names
all_tensors = graph.tensors()

# Find nodes that are only in the post-processing chain
nodes_to_remove = []
keep_nodes = set()

# Walk backwards from outputs
output_tensor = [t for t in graph.tensors().values() if t.name == "output0"][0]

# Mark nodes that are part of post-processing (after Split_1)
postproc_tensors = set()
postproc_tensors.add(output_tensor.name)

# Do BFS from output back to (but not including) Split_1
queue = [output_tensor]
visited = set()

while queue:
    t = queue.pop(0)
    if t.name in visited:
        continue
    visited.add(t.name)
    
    if t.name == "/model.23/Split_1_output_0" or t.name == "/model.23/Split_1_output_1":
        continue  # Stop at Split_1 outputs
    
    if t.inputs:
        for producer in t.inputs:
            if producer:
                nodes_to_remove.append(producer)
                for inp in producer.inputs:
                    if inp.name not in visited:
                        queue.append(inp)

# Also find all nodes between Split_1 and the output
all_nodes_prenames = set(n.name for n in graph.nodes)
nodes_names = set()

# Collect all downstream nodes from Split_1 outputs
for tensor_name in ["/model.23/Split_1_output_0", "/model.23/Split_1_output_1"]:
    t = all_tensors[tensor_name]
    for consumer in t.outputs:
        _collect_downstream(consumer, nodes_names, all_tensors)

def _collect_downstream(node, collected, tensors):
    if node.name in collected:
        return
    collected.add(node.name)
    for out in node.outputs:
        if out.name in tensors:  # it's a registered tensor
            for consumer in out.outputs:
                _collect_downstream(consumer, collected, tensors)

# Collect
downstream_names = set()
for tensor_name in ["/model.23/Split_1_output_0", "/model.23/Split_1_output_1"]:
    t = all_tensors[tensor_name]
    for consumer in t.outputs:
        _collect_downstream(consumer, downstream_names, all_tensors)

print(f"\nNodes downstream of Split_1 (to remove): {len(downstream_names)}")

# Remove all downstream nodes
graph.nodes = [n for n in graph.nodes if n.name not in downstream_names]

# Set graph outputs to the Split_1 outputs (raw bbox + scores)
# We want the concatenated format [batch, 8400, 84] = [batch, 8400, 4+80]
graph.outputs = [transpose]
transpose.name = "raw_detections"

# Clean up
graph.cleanup().toposort()

# Fix output metadata
for out in graph.outputs:
    out.dtype = np.float32

# Save
onnx_model = gs.export_onnx(graph)
onnx.save(onnx_model, "yolo26n_raw.onnx")

# Verify
model2 = onnx.load("yolo26n_raw.onnx")
onehot_count = len([n for n in model2.graph.node if n.op_type == "OneHot"])
print(f"\nPatched model: {len(model2.graph.node)} nodes, {len(model2.graph.output)} outputs")
print(f"OneHot remaining: {onehot_count}")
print(f"Outputs: {[o.name for o in model2.graph.output]}")

import os
print(f"Size: {os.path.getsize('yolo26n_raw.onnx')/1024/1024:.1f} MB")
