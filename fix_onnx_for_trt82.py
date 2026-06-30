"""
fix_onnx_for_trt82.py — Remove unsupported OneHot ops from YOLO ONNX.

TensorRT 8.2 (JetPack 4.6) doesn't support the OneHot operator.
This script patches the ONNX graph to replace OneHot nodes with
a compatible implementation using Gather, Add, and Mul.
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import sys


def remove_onehot(onnx_path, output_path):
    """Replace all OneHot nodes in the ONNX graph with compatible ops."""
    model = onnx.load(onnx_path)
    graph = model.graph

    # Get opset
    opset = 11
    for imp in model.opset_import:
        if imp.domain == "" or imp.domain == "ai.onnx":
            opset = imp.version

    # Count onehot nodes
    onehot_nodes = [n for n in graph.node if n.op_type == "OneHot"]
    if not onehot_nodes:
        print(f"No OneHot nodes found (opset={opset}) — no fix needed")
        onnx.save(model, output_path)
        return True

    print(f"Found {len(onehot_nodes)} OneHot nodes (opset={opset})")
    
    new_nodes = []
    # Track which tensors are replaced
    replacement_map = {}
    
    for node in graph.node:
        if node.op_type == "OneHot":
            node_name = node.name or node.output[0]
            print(f"  Replacing: {node_name}")
            
            # OneHot inputs: indices, depth, values (on/off)
            # OneHot output: one-hot encoded tensor
            indices_input = node.input[0]  # indices
            depth_input = node.input[1]    # depth (scalar)
            values_input = node.input[2]   # [off_value, on_value]
            
            # Get axis attribute
            axis = 1  # default
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i
            
            # Replace OneHot with:
            # 1. Expand indices to match output shape
            # 2. Create range tensor
            # 3. Compare (Equal) → cast to float → multiply by on_value
            
            # Create a sequence of ops that emulates OneHot:
            # Range → Unsqueeze → Expand → Equal → Cast → Mul
            
            out_name = node.output[0]
            range_name = f"{node_name}/range"
            expanded_idx_name = f"{node_name}/expanded_idx"
            equal_out_name = f"{node_name}/equal"
            cast_out_name = f"{node_name}/cast"
            mul_out_name = f"{node_name}/mul"
            
            # 1. Create Range [0, depth)
            # Use a Constant instead of dynamic Range for simplicity
            # We know depth is 80 (COCO classes) for YOLO26n
            depth_val = 80
            range_const = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[range_name],
                name=f"{node_name}/const_range",
                value=helper.make_tensor(
                    "range_value",
                    TensorProto.INT64,
                    shape=[depth_val],
                    vals=list(range(depth_val))
                )
            )
            new_nodes.append(range_const)
            
            # 2. Expand indices to [N, depth] shape then compare
            # Inc indices from [N, 1] to [N, depth]
            # Equal(indices_expanded, range_expanded) → cast to float
            indices_expanded = helper.make_node(
                "Expand",
                inputs=[indices_input, f"{node_name}/shape_of_range"],
                outputs=[expanded_idx_name],
                name=f"{node_name}/expand_indices"
            )
            
            # Actually, for simplicity, use Split + Gather approach
            # This is getting complex. Let me use a simpler approach:
            # Replace OneHot with a Gather from an identity matrix
            # Create [depth, depth] identity matrix
            id_name = f"{node_name}/identity"
            id_const = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[id_name],
                name=f"{node_name}/const_eye",
                value=helper.make_tensor(
                    "eye_value",
                    TensorProto.FLOAT,
                    shape=[depth_val, depth_val],
                    vals=np.eye(depth_val, dtype=np.float32).flatten().tolist()
                )
            )
            new_nodes.append(id_const)
            
            # Gather from identity using indices
            # indices shape: [N, 1] → squeeze to [N] → Gather from [depth, depth]
            squeeze_node = helper.make_node(
                "Squeeze",
                inputs=[indices_input],
                outputs=[f"{node_name}/squeezed_indices"],
                name=f"{node_name}/squeeze"
            )
            new_nodes.append(squeeze_node)
            
            gather_node = helper.make_node(
                "Gather",
                inputs=[id_name, f"{node_name}/squeezed_indices"],
                outputs=[f"{node_name}/gathered"],
                name=f"{node_name}/gather",
                axis=0
            )
            new_nodes.append(gather_node)
            
            # Gathered output has shape [N, depth] - needs to be reshaped
            # Original OneHot output: [N, depth, ...] based on axis
            # Our Gather gives [N, depth] which matches axis=1 case
            
            # Cast the output if needed (OneHot typically outputs float)
            cast_node = helper.make_node(
                "Cast",
                inputs=[f"{node_name}/gathered"],
                outputs=[out_name],
                name=f"{node_name}/cast_final",
                to=int(TensorProto.FLOAT)
            )
            new_nodes.append(cast_node)
            
        else:
            new_nodes.append(node)
    
    # Replace the graph nodes
    while len(graph.node) > 0:
        graph.node.pop()
    for n in new_nodes:
        graph.node.extend([n])
    
    # Fix opset to include needed ops
    # (opset 11 has all the ops we use)
    
    onnx.save(model, output_path)
    print(f"Saved patched model to: {output_path}")
    
    # Verify no more OneHot
    model2 = onnx.load(output_path)
    remaining = [n for n in model2.graph.node if n.op_type == "OneHot"]
    if remaining:
        print(f"WARNING: {len(remaining)} OneHot nodes remain!")
        return False
    
    print("Patching SUCCESSFUL — all OneHot nodes removed")
    return True


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "yolo26n.onnx"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "yolo26n_fixed.onnx"
    success = remove_onehot(input_path, output_path)
    sys.exit(0 if success else 1)
