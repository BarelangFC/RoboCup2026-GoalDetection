import onnx, sys
m = onnx.load(sys.argv[1])
total = len(m.graph.node)
print(f"Total nodes: {total}")
# Print the last 50 nodes
for i in range(max(0, total-50), total):
    n = m.graph.node[i]
    out_str = ", ".join(o[:30] for o in n.output)
    print(f"[{i}] {n.op_type}: {out_str}")
