
import onnx
import onnx_graphsurgeon as gs
import numpy as np

# Load model
graph = gs.import_onnx(onnx.load("yolo26n.onnx"))
print(f"Original: {len(graph.nodes)} nodes, {len(graph.outputs)} outputs")
print("Outputs:", [o.name for o in graph.outputs])

# Find all OneHot nodes
onehot_nodes = [n for n in graph.nodes if n.op == "OneHot"]
print(f"\nOneHot nodes: {len(onehot_nodes)}")

# For each OneHot, trace back to find the tensor BEFORE the OneHot processing
# The OneHot is part of NMS/selection post-processing
# We want to output the raw detection tensors instead

# Find the node(s) that feed into OneHot
for oh in onehot_nodes:
    print(f"\nOneHot: {oh.name}")
    print(f"  Inputs: {[i.name for i in oh.inputs]}")
    print(f"  Outputs: {[o.name for o in oh.outputs]}")
    
    for inp in oh.inputs:
        producer = inp.inputs[0] if inp.inputs else None
        if producer:
            print(f"  Input producer: {producer.name} ({producer.op})")

# Find all outputs
for o in graph.outputs:
    producer = o.inputs[0] if o.inputs else None
    if producer:
        print(f"\nOutput: {o.name}, producer: {producer.name} ({producer.op})")

# The cleanest approach: find the /model.23/Split_1 output which contains
# [bboxes (4), class_scores (80)] concatenated and expose those as graph outputs

# Look for the Split node after /model.23/Transpose
for n in graph.nodes:
    if n.op == "Split" and "model.23" in n.name:
        print(f"\nSplit node: {n.name}")
        print(f"  Inputs: {[i.name for i in n.inputs]}")
        print(f"  Outputs: {[o.name for o in n.outputs]}")

# Find the original detection tensor (before any post-processing)
# It's the output of /model.23/Concat or /model.23/Transpose
for n in graph.nodes:
    if n.op == "Transpose" and "model.23" in n.name:
        print(f"\nTranspose: {n.name}")
        print(f"  Inputs: {[i.name for i in n.inputs]}")
        print(f"  Outputs: {[o.name for o in n.outputs]}")

# Also find the /model.23/Concat
for n in graph.nodes:
    if n.op == "Concat" and "model.23" in n.name:
        print(f"\nConcat: {n.name}")
        print(f"  Inputs: {[i.name for i in n.inputs]}")
        print(f"  Outputs: {[o.name for o in n.outputs]}")
