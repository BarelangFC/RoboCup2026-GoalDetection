import onnx, sys

m = onnx.load(sys.argv[1])
print(f"IR: {m.ir_version}")
for s in m.opset_import:
    print(f"  Domain: {s.domain}, Version: {s.version}")
print(f"Nodes: {len(m.graph.node)}, Outputs: {len(m.graph.output)}")

# Outputs
for o in m.graph.output:
    t = o.type.tensor_type
    shape = [d.dim_value for d in t.shape.dim]
    print(f"  Output: {o.name} shape={shape}")

# OneHot
onehot = [n for n in m.graph.node if n.op_type == "OneHot"]
print(f"OneHot: {len(onehot)}")

# Transpose in model.23
for n in m.graph.node:
    if n.op_type == "Transpose":
        print(f"  Transpose: {n.name} -> {n.output}")

# Class count from final layers
print("\nLast 10 nodes:")
for n in m.graph.node[-10:]:
    print(f"  {n.op_type}: {n.name}")

# Check all unique ops
ops = set()
for n in m.graph.node:
    ops.add(n.op_type)
print(f"\nUnique ops ({len(ops)}): {sorted(ops)}")
