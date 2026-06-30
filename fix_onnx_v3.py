import onnx
import onnx_graphsurgeon as gs
import numpy as np
import os

graph = gs.import_onnx(onnx.load("yolo26n.onnx"))
all_tensors = graph.tensors()

transpose_out = all_tensors["/model.23/Transpose_output_0"]

# Set dtype on the output tensor
transpose_out.dtype = np.float32
transpose_out.shape = (1, 8400, 84)

def collect_downstream(start_tensor):
    collected = set()
    queue = list(start_tensor.outputs)
    while queue:
        node = queue.pop(0)
        if node.name in collected:
            continue
        collected.add(node.name)
        for out_t in node.outputs:
            for consumer in out_t.outputs:
                queue.append(consumer)
    return collected

to_remove = collect_downstream(transpose_out)
graph.nodes = [n for n in graph.nodes if n.name not in to_remove]
graph.outputs = [transpose_out]
transpose_out.name = "raw_detections"

graph.cleanup().toposort()

onnx_model = gs.export_onnx(graph)
onnx.save(onnx_model, "yolo26n_raw.onnx")

model2 = onnx.load("yolo26n_raw.onnx")
onehot_count = len([n for n in model2.graph.node if n.op_type == "OneHot"])
print(f"Nodes: {len(model2.graph.node)}, Outputs: {len(model2.graph.output)}")
print(f"OneHot: {onehot_count}")
print(f"Size: {os.path.getsize('yolo26n_raw.onnx')/1024/1024:.1f} MB")
for o in model2.graph.output:
    print(f"  Output: {o.name}")