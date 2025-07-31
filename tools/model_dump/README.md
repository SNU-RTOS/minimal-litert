# TFLite Model Structure Analyzer

## Overview

This tool is a Python script designed to parse TensorFlow Lite (`.tflite`) models and generate a detailed, human-readable log of their internal structure. It is particularly useful for understanding complex models that utilize nested subgraphs, such as those containing `STABLEHLO_COMPOSITE` operators.

## Features

- **Hierarchical Structure Parsing:** Recursively parses the model's subgraphs and operators to reveal its complete structure.
- **`STABLEHLO_COMPOSITE` Support:** Intelligently handles `STABLEHLO_COMPOSITE` operators by recursively analyzing their decomposed subgraphs.
- **Detailed Node Analytics:** For each subgraph, it provides:
  - Total number of operators.
  - Total number of operators including all nested subgraphs.
  - Count of `STABLEHLO_COMPOSITE` operators.
- **Interactive CLI:** An easy-to-use command-line interface allows you to choose between:
  1. Parsing the entire model.
  2. Visualizing a single, specific subgraph.

## Setup

The parser requires Python bindings for the TFLite schema, which can be generated from `schema.fbs` using the FlatBuffers compiler (`flatc`).

A helper script, `parse_model.sh`, is provided to automate this process.

1. **Ensure you have the FlatBuffers compiler (`flatc`)**. The script expects `flatc_x64` (for x86_64) or `flatc_arm64` (for aarch64) in the current directory.
2. Run the `parse_model.sh` script. It will automatically check for the generated schema files and create them if they are missing.

   ```bash
   ./parse_model.sh
   ```

## Usage

You can run the parser directly or use the provided shell script.

### 1. Running with the script (Recommended)

Modify `parse_model.sh` to point to the desired model file and execute it.

```bash
# In parse_model.sh
# ...
python3 parser.py -m ../../models/Llama3.2-3B/llama3.2_q8_ekv1024.tflite
```

Then run the script:

```bash
./parse_model.sh
```

### 2. Running Manually

Execute the Python script directly, optionally providing a path to your model.

```bash
python3 parser.py -m /path/to/your/model.tflite
```

If no model is specified, it uses a default path.

### Interactive Modes

After launching, the script will display the model being parsed and prompt you to choose a parsing mode:

- **`1. Entire model`**: Parses every subgraph in the model and saves the output to a single `_entire.log` file.
- **`2. Subgraph visualization`**: Lists all available signature definitions (e.g., `prefill`, `decode`) and their corresponding subgraph indices. You can then select a specific subgraph to parse. The output is saved to a log file named after the selected signature.

## Output Format

The script generates a `.log` file that details the model structure. Here is a sample output for a single subgraph:

```log
Subgraph 6: prefill (Total Nodes: 87, Including Nested: 537, STABLEHLO_COMPOSITE: 1)
  [0000] RESHAPE
  [0001] STABLEHLO_COMPOSITE  (→ Subgraph 538: odml.rms_norm.impl_355)
   --------------------------------
    Subgraph 538: odml.rms_norm.impl_355 (Total Nodes: 5, Including Nested: 5, STABLEHLO_COMPOSITE: 0)
      [0000] CAST
      [0001] MULTIPLY
      ...
   --------------------------------
  [0002] CONCATENATION
  ...
```

## Future Plans

- [] Support Tensor info dump
