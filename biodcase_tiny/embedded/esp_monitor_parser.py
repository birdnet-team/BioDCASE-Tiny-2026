import re
import yaml
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from pathlib import Path
from datetime import datetime

# Model location config
MODEL_DIRS  = [Path("./output/03_models")]
TFLITE_MODEL_EXT = '.tflite'
OUTPUT_DIRS = [Path("./output/04_reports"), Path("./reports")]


def _find_tflite() -> Path | None:
    """Return the first matching .tflite file across all known dirs and names."""
    for folder in MODEL_DIRS:

        # candidate files 
        candidate_model_files = sorted(list(folder.glob('*' + TFLITE_MODEL_EXT)))

        # nothing found
        if not len(candidate_model_files): continue

        # more than one file
        #if len(candidate_model_files) != 1: print("***More than one .tflite file, take first")

        # just take first model file
        return Path(candidate_model_files[0])

    return None


def _find_output_dir() -> Path | None:
    """Return the first existing output directory across all known options."""
    for folder in OUTPUT_DIRS:
        if folder.exists():
            return folder
    return None


def get_conv2d_macs(input_shapes, output_shapes):
    """Compute MACs for a CONV_2D layer."""
    _, out_h, out_w, out_ch = output_shapes[0]
    out_ch_k, k_h, k_w, in_ch = input_shapes[1]
    return out_h * out_w * k_h * k_w * in_ch * out_ch


def get_depthwise_conv2d_macs(input_shapes, output_shapes):
    """Compute MACs for a DEPTHWISE_CONV_2D layer."""
    _, out_h, out_w, out_ch = output_shapes[0]
    _, k_h, k_w, _ = input_shapes[1]
    return out_h * out_w * k_h * k_w * out_ch


def get_fully_connected_macs(input_shapes, output_shapes):
    """Compute MACs for a FULLY_CONNECTED (dense) layer."""
    weight_shape = input_shapes[1]      # [out_features, in_features]
    out_features = weight_shape[0]
    in_features  = weight_shape[1]
    return in_features * out_features


def get_transpose_conv2d_macs(input_shapes, output_shapes):
    """Compute MACs for a TRANSPOSE_CONV layer."""
    _, out_h, out_w, out_ch = output_shapes[0]
    _, k_h, k_w, in_ch = input_shapes[1]
    return out_h * out_w * k_h * k_w * in_ch * out_ch


def get_pool2d_macs(input_shapes, output_shapes):
    """Compute MACs for AVG/MAX pool layers (comparisons counted as MACs)."""
    _, out_h, out_w, out_ch = output_shapes[0]
    _, in_h, in_w, _ = input_shapes[0]
    # kernel size inferred from input/output ratio
    k_h = in_h // out_h if out_h > 0 else 1
    k_w = in_w // out_w if out_w > 0 else 1
    return out_h * out_w * k_h * k_w * out_ch


def get_elementwise_macs(output_shapes):
    """Compute MACs for elementwise ops like ADD, MUL, SUB."""
    shape = output_shapes[0]
    result = 1
    for dim in shape:
        result *= dim
    return result


def compute_macs(model_path):
    """
    Estimate the total MAC (multiply-accumulate) count for a TFLite model.
    Note: This is an estimate based on common TFLite ops and may not be exact.
    The result is best used for relative comparisons between models or layers.
    """
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    # Build a lookup dict for tensor details by tensor index
    tensor_details = {t['index']: t for t in interpreter.get_tensor_details()}

    total_macs = 0
    results = []
    skipped_ops = set()

    for op in interpreter._get_ops_details():
        op_name = op['op_name']

        input_shapes = []
        for t in op['inputs']:
            if t >= 0 and t in tensor_details:
                input_shapes.append(tensor_details[t]['shape'].tolist())

        output_shapes = []
        for t in op['outputs']:
            if t >= 0 and t in tensor_details:
                output_shapes.append(tensor_details[t]['shape'].tolist())

        macs = 0
        try:
            if op_name == 'CONV_2D':
                macs = get_conv2d_macs(input_shapes, output_shapes)
            elif op_name == 'DEPTHWISE_CONV_2D':
                macs = get_depthwise_conv2d_macs(input_shapes, output_shapes)
            elif op_name == 'FULLY_CONNECTED':
                macs = get_fully_connected_macs(input_shapes, output_shapes)
            elif op_name == 'TRANSPOSE_CONV':
                macs = get_transpose_conv2d_macs(input_shapes, output_shapes)
            elif op_name in ('AVERAGE_POOL_2D', 'MAX_POOL_2D'):
                macs = get_pool2d_macs(input_shapes, output_shapes)
            elif op_name in ('ADD', 'MUL', 'SUB'):
                macs = get_elementwise_macs(output_shapes)
            else:
                skipped_ops.add(op_name)
        except (IndexError, ValueError) as e:
            print(f"  Warning: could not compute MACs for {op_name}: {e}")

        if macs > 0:
            results.append((op_name, macs))
            total_macs += macs

    if skipped_ops:
        print(f"Skipped ops (MACs not counted): {sorted(skipped_ops)}")

    return total_macs

def parse_monitor_output(lines: list[str], report_dir: Path = Path(".")) -> dict:
    """
    Parse ESP32 serial monitor output and write a YAML report.

    Extracts:
      - model size (bytes) from .tflite file found via MODEL_DIRS / *.TFLITE_MODEL_EXT
      - Estimated MACs (multiply-accumulate operations) from the .tflite file
      - setup time (µs)         - block 1: GetFeatureConfig, AllocateTensors, etc.
      - preprocessing time (µs) - block 2: FFT, Mel filterbank, etc.
      - inference time (µs)     - block 3: CONV, DEPTHWISE_CONV, FULLY_CONNECTED, etc.
      - total time (µs)         - sum of all three
      - RAM arena allocation (total, head, tail)
      - CRC32 checksums (reproducibility checks)
    """

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_size_bytes": None,
        "Estimated MACs": None,
        "timing_us": {
            "setup": None,
            "preprocessing": None,
            "inference": None,
            "total": None,
        },
        "ram_bytes": {
            "arena_total": None,
            "arena_head": None,
            "arena_tail": None,
        },
        "crc32": {
            "audio_input": None,
            "features_output": None,
            "model_input": None,
            "model_output": None,
        },   
    }

    # Model size
    tflite = _find_tflite()
    if tflite:
        result["model_size_bytes"] = tflite.stat().st_size
        try:
            result["Estimated MACs"] = compute_macs(tflite)
        except Exception as e:
            print(f"[esp_monitor_parser] Error computing MACs: {e}")
    else:
        print(f"[esp_monitor_parser] Warning: no {TFLITE_MODEL_EXT} model found in {MODEL_DIRS}")

    # The log contains three CSV timing tables:
    #   1st = model setup/init  (GetFeatureConfig, AllocateTensors, ...)
    #   2nd = feature extraction (FFT, Mel, ...)
    #   3rd = TFLite ops / inference (CONV, DEPTHWISE_CONV, ...)
    timing_block_index = 0
    in_timing_block = False

    for line in lines:

        # Timing blocks (CSV tables)
        if '"Unique Tag","Total microseconds across all events with that tag."' in line:
            timing_block_index += 1
            in_timing_block = True
            continue

        if in_timing_block:
            m = re.match(r'"total number of microseconds",\s*(\d+)', line)
            if m:
                us = int(m.group(1))
                if timing_block_index == 1:
                    result["timing_us"]["setup"] = us
                elif timing_block_index == 2:
                    result["timing_us"]["preprocessing"] = us
                elif timing_block_index == 3:
                    result["timing_us"]["inference"] = us
                in_timing_block = False
                continue

        # CRC32 checksums
        m = re.match(r'Audio Input CRC32:\s*(0x[0-9A-Fa-f]+)', line)
        if m:
            result["crc32"]["audio_input"] = m.group(1)
            continue

        m = re.match(r'Output Features CRC32:\s*(0x[0-9A-Fa-f]+)', line)
        if m:
            result["crc32"]["features_output"] = m.group(1)
            continue

        m = re.match(r'Input CRC32:\s*(0x[0-9A-Fa-f]+)', line)
        if m:
            result["crc32"]["model_input"] = m.group(1)
            continue

        m = re.match(r'Output CRC32:\s*(0x[0-9A-Fa-f]+)', line)
        if m:
            result["crc32"]["model_output"] = m.group(1)
            continue

        # RAM / arena allocation
        m = re.match(r'\[RecordingMicroAllocator\] Arena allocation total\s+(\d+)\s+bytes', line)
        if m:
            result["ram_bytes"]["arena_total"] = int(m.group(1))
            continue

        m = re.match(r'\[RecordingMicroAllocator\] Arena allocation head\s+(\d+)\s+bytes', line)
        if m:
            result["ram_bytes"]["arena_head"] = int(m.group(1))
            continue

        m = re.match(r'\[RecordingMicroAllocator\] Arena allocation tail\s+(\d+)\s+bytes', line)
        if m:
            result["ram_bytes"]["arena_tail"] = int(m.group(1))
            continue

    # Derived total time
    times = [result["timing_us"][k] for k in ("setup", "preprocessing", "inference")]
    if all(t is not None for t in times):
        result["timing_us"]["total"] = sum(times)

    # Write YAML
    output_dir = _find_output_dir() or report_dir
    report_path = Path(output_dir) / "monitor_report.yaml"
    with open(report_path, "w") as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)

    print(f"[esp_monitor_parser] Report written to {report_path}")
    return result