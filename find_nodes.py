import onnx, sys
m = onnx.load(sys.argv[1])

# Find nodes after the raw bbox decoding (Reshape_7, Unsqueeze_1)
targets = ["Reshape_7", "Unsqueeze_1", "Unsqueeze_2", "Mod", "Concat_6"]
for i, n in enumerate(m.graph.node):
    name = n.name.split("/")[-1]
    if name in targets or any(t in name for t in targets):
        print(f"[{i}] {n.op_type}: {n.name}")
        for o in n.output:
            print(f"  -> {o[:60]}")

# Also find Concat_3 output (before DFL) and Concat_6 output (final)
for i, n in enumerate(m.graph.node):
    if n.op_type == "Concat":
        name = n.name.split("/")[-1]
        if name in ["Concat_3", "Concat_4", "Concat_5", "Concat_6"]:
            print(f"[{i}] {n.op_type}: {n.name} -> {[o[:40] for o in n.output]}")
