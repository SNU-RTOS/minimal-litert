import flatbuffers
from tflite.Model import Model
from tflite.OperatorCode import OperatorCode
from tflite.BuiltinOperator import BuiltinOperator
from tflite.BuiltinOptions import BuiltinOptions
from tflite.BuiltinOptions2 import BuiltinOptions2
from tflite.Operator import Operator
from tflite.StableHLOCompositeOptions import StableHLOCompositeOptions
import argparse
import os
from pathlib import Path

# from tensorflow.lite.python.interpreter import Interpreter
import tensorflow as tf



# === Builtin Operator Reverse Mapping (enum 값 → 이름)
BUILTIN_OPERATOR_NAME = {
    v: k
    for k, v in vars(BuiltinOperator).items()
    if not k.startswith("__") and isinstance(v, int)
}

# === Helper: Get opcode name ===
def get_opcode_name(opcode: OperatorCode) -> str:
    builtin = opcode.BuiltinCode()
    if builtin == BuiltinOperator.CUSTOM:
        return opcode.CustomCode().decode("utf-8")
    return BUILTIN_OPERATOR_NAME.get(builtin, f"UNKNOWN_BUILTIN_{builtin}")


# === Parse and log specific subgraph only ===
def parse_single_subgraph(model, opcodes, subgraphs, subgraph_index, log):
    # Removed the `open` call since `log` is now a file object passed by the caller
    parse_recursive_subgraph(model, opcodes, subgraphs, subgraph_index, log)


# === Recursive parsing of subgraph and STABLEHLO_COMPOSITE ===
def parse_recursive_subgraph(model, opcodes, subgraphs, subgraph_index, log, depth=0):
    subgraph = subgraphs[subgraph_index]
    name = subgraph.Name().decode("utf-8") if subgraph.Name() else f"(unnamed_{subgraph_index})"
    indent = "  " * depth
    total_nodes = subgraph.OperatorsLength()
    stablehlo_count = sum(
        1 for i in range(subgraph.OperatorsLength())
        if get_opcode_name(opcodes[subgraph.Operators(i).OpcodeIndex()]) == "STABLEHLO_COMPOSITE"
    )
    
    def count_all_nodes(subgraph_index, depth=0):
        subgraph = subgraphs[subgraph_index]
        total = subgraph.OperatorsLength()

        for i in range(subgraph.OperatorsLength()):
            op: Operator = subgraph.Operators(i)
            opcode_idx = op.OpcodeIndex()
            opcode = opcodes[opcode_idx]
            op_name = get_opcode_name(opcode)

            if op_name == "STABLEHLO_COMPOSITE":
                if op.BuiltinOptions2Type() == BuiltinOptions2.StableHLOCompositeOptions:
                    stablehlo_opts = StableHLOCompositeOptions()
                    stablehlo_opts.Init(op.BuiltinOptions2().Bytes, op.BuiltinOptions2().Pos)
                    child_subgraph_idx = stablehlo_opts.DecompositionSubgraphIndex()
                    total += count_all_nodes(child_subgraph_idx, depth + 1)

        return total

    total_nodes_including_nested = count_all_nodes(subgraph_index)
    log.write(f"{indent}Subgraph {subgraph_index}: {name} (Total Nodes: {total_nodes}, Including Nested: {total_nodes_including_nested}, STABLEHLO_COMPOSITE: {stablehlo_count})\n")

    for i in range(subgraph.OperatorsLength()):
        op: Operator = subgraph.Operators(i)
        opcode_idx = op.OpcodeIndex()
        opcode = opcodes[opcode_idx]
        op_name = get_opcode_name(opcode)

        line = f"{indent}  [{i:04}] {op_name}"

        if op_name == "STABLEHLO_COMPOSITE":
            if op.BuiltinOptions2Type() == BuiltinOptions2.StableHLOCompositeOptions:
                stablehlo_opts = StableHLOCompositeOptions()
                stablehlo_opts.Init(op.BuiltinOptions2().Bytes, op.BuiltinOptions2().Pos)
                child_subgraph_idx = stablehlo_opts.DecompositionSubgraphIndex()
                child_subgraph_name = subgraphs[child_subgraph_idx].Name().decode("utf-8") if subgraphs[child_subgraph_idx].Name() else f"(unnamed_{child_subgraph_idx})"
                log.write(line + f"  (→ Subgraph {child_subgraph_idx}: {child_subgraph_name})\n")
                log.write(f"   --------------------------------\n")
                parse_recursive_subgraph(model, opcodes, subgraphs, child_subgraph_idx, log, depth + 2)
                log.write(f"   --------------------------------\n")
                continue

        log.write(line + "\n")
        
# === Parse and log entire TFLite model structure ===
def parse_entire_tflite_model(model_file, log_file):
    # Load TFLite FlatBuffer Model
    with open(model_file, "rb") as f:
        buf = f.read()

    model = Model.GetRootAs(buf, 0)
    opcodes = [model.OperatorCodes(i) for i in range(model.OperatorCodesLength())]
    subgraphs = [model.Subgraphs(i) for i in range(model.SubgraphsLength())]

    with open(log_file, "w") as log:
        for subgraph_index in range(len(subgraphs)):
            parse_single_subgraph(model, opcodes, subgraphs, subgraph_index, log)
            log.write("\n\n")


def parse_runtime_tflite_model(model_file):
    
    interpreter = tf.lite.Interpreter(
        model_path=model_file,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.AUTO
    )
    interpreter.allocate_tensors()

    ops = interpreter._get_ops_details()
    tensors = interpreter.get_tensor_details()

    for op in ops:
        print(f"[{op['index']:02}] {op['op_name']}")
        # for tidx in op['inputs']:
        #     t = tensors[tidx]
        #     print(f"    input: {t['name']} ({t['shape']})")
        # for tidx in op['outputs']:
        #     t = tensors[tidx]
        #     print(f"    output: {t['name']} ({t['shape']})")

# === Main CLI Entrypoint ===
def main():
    parser = argparse.ArgumentParser(description="Parse and log TFLite model structure.")
    parser.add_argument(
        "-m", "--model_file",
        type=str,
        help="Path to the TFLite model file.",
        default="../../models/mobileone_s0.tflite",
    )
    args = parser.parse_args()

    # Validate model file path
    print(f"\nParsing model: {args.model_file}")
    model_path = Path(args.model_file)
    if not model_path.is_file():
        print(f"Error: Model file '{args.model_file}' does not exist.")
        return

    print("Choose parsing mode:")
    print("1. Entire model")
    print("2. Subgraph visualization")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        log_file = f"{os.path.splitext(args.model_file)[0]}_entire.log"
        parse_entire_tflite_model(args.model_file, log_file)
        print(f"\n✔ Parsing complete. Log saved to: {log_file}")

    elif choice == "2":
        with open(args.model_file, "rb") as f:
            buf = f.read()
        model = Model.GetRootAs(buf, 0)
        opcodes = [model.OperatorCodes(i) for i in range(model.OperatorCodesLength())]
        subgraphs = [model.Subgraphs(i) for i in range(model.SubgraphsLength())]

        print("\n=== Available Signature Names ===")
        for i in range(model.SignatureDefsLength()):
            sig = model.SignatureDefs(i)
            name = sig.SignatureKey().decode("utf-8") if sig.SignatureKey() else f"(unnamed_{i})"
            subgraph_idx = sig.SubgraphIndex()
            print(f"  [{i}] {name} → subgraph index: {subgraph_idx}")

        try:
            selected_index = int(input("\nEnter subgraph index to visualize: "))
            assert 0 <= selected_index < len(subgraphs)
        except Exception as e:
            print("Invalid subgraph index.", e)
            return

        log_file_path = f"{os.path.splitext(args.model_file)[0]}_{model.SignatureDefs(selected_index).SignatureKey().decode('utf-8')}.log"
        with open(log_file_path, "w") as log_file:
            parse_single_subgraph(model, opcodes, subgraphs, selected_index, log_file)
        print(f"\n✔ Log output saved to {log_file_path}")

    else:
        print("Invalid choice. Exiting.")
        exit(0)


    # parse_runtime_tflite_model(args.model_file)
    # tf.lite.experimental.Analyzer.analyze(model_path=args.model_file, 
    #                                       gpu_compatibility=True,
    #                                     #   experimental_use_mlir=True
    #                                       )

if __name__ == "__main__":
    main()


# === End of file: dev/app/minimal-litert/tools/model_dump/parser.py ===

