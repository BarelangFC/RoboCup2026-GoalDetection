import onnx, sys
m = onnx.load(sys.argv[1])
# Find model.23 section
found_transpose = False
for i, n in enumerate(m.graph.node):
    if "23" in n.name and n.op_type in ["Transpose", "Split", "Softmax", "Mul", "Sigmoid", "Conv"]:
        if not found_transpose and n.op_type == "Transpose":
            found_transpose = True
        if found_transpose:
            print(f"[{i}] {n.op_type}: {n.name.split('/')[-1][:40]}")
            if i > 500: break
