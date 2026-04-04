import re
import yaml
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from pathlib import Path
from datetime import datetime


# constants

TFLITE_MODEL_EXT = ".tflite"

FRAMEWORK_MODEL_DIRS = {
    "tensorflow": Path("./output/03_models/tensorflow"),
    "pytorch":    Path("./output/03_models/pytorch"),
}

REPORT_DIR  = Path("./output/04_reports")
REPORT_FILE = REPORT_DIR / "monitor_report.yaml"

FRAMEWORK_REPORT_DIRS = {
    "tensorflow": REPORT_DIR / "tensorflow",
    "pytorch":    REPORT_DIR / "pytorch",
}


# mac estimation functions

def get_conv2d_macs(input_shapes, output_shapes):
    """Compute MACs for a CONV_2D layer."""
    _, out_h, out_w, out_ch = output_shapes[0]
    _, k_h, k_w, in_ch = input_shapes[1]
    return out_h * out_w * k_h * k_w * in_ch * out_ch

def get_depthwise_conv2d_macs(input_shapes, output_shapes):
    """Compute MACs for a DEPTHWISE_CONV_2D layer."""
    _, out_h, out_w, out_ch = output_shapes[0]
    _, k_h, k_w, _ = input_shapes[1]
    return out_h * out_w * k_h * k_w * out_ch

def get_fully_connected_macs(input_shapes, output_shapes):
    """Compute MACs for a FULLY_CONNECTED (dense) layer."""
    weight_shape = input_shapes[1]
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
    k_h = in_h // out_h if out_h > 0 else 1
    k_w = in_w // out_w if out_w > 0 else 1
    return out_h * out_w * k_h * k_w * out_ch

def get_elementwise_macs(output_shapes):
    """Compute MACs for elementwise ops like ADD, MUL, SUB."""
    result = 1
    for dim in output_shapes[0]:
        result *= dim
    return result

def compute_macs(model_path: Path) -> int:
    """
    Estimate the total MAC count for a TFLite model.

    Note: This is an estimate based on common TFLite ops and may not be exact.
    The result is best used for relative comparisons between models or layers.
    """
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    tensor_details = {t["index"]: t for t in interpreter.get_tensor_details()}

    total_macs = 0
    skipped_ops: set[str] = set()

    for op in interpreter._get_ops_details():
        op_name = op["op_name"]

        input_shapes = [
            tensor_details[t]["shape"].tolist()
            for t in op["inputs"]
            if t >= 0 and t in tensor_details
        ]
        output_shapes = [
            tensor_details[t]["shape"].tolist()
            for t in op["outputs"]
            if t >= 0 and t in tensor_details
        ]

        macs = 0
        try:
            if op_name == "CONV_2D":
                macs = get_conv2d_macs(input_shapes, output_shapes)
            elif op_name == "DEPTHWISE_CONV_2D":
                macs = get_depthwise_conv2d_macs(input_shapes, output_shapes)
            elif op_name == "FULLY_CONNECTED":
                macs = get_fully_connected_macs(input_shapes, output_shapes)
            elif op_name == "TRANSPOSE_CONV":
                macs = get_transpose_conv2d_macs(input_shapes, output_shapes)
            elif op_name in ("AVERAGE_POOL_2D", "MAX_POOL_2D"):
                macs = get_pool2d_macs(input_shapes, output_shapes)
            elif op_name in ("ADD", "MUL", "SUB"):
                macs = get_elementwise_macs(output_shapes)
            else:
                skipped_ops.add(op_name)
        except (IndexError, ValueError) as e:
            print(f"  Warning: could not compute MACs for {op_name}: {e}")

        total_macs += macs

    if skipped_ops:
        print(f"Skipped ops (MACs not counted): {sorted(skipped_ops)}")

    return total_macs


# internal helper functions

def _find_tflite_in(folder: Path) -> Path | None:
    """Return the first .tflite file found in *folder*, or None."""
    candidates = sorted(folder.glob("*" + TFLITE_MODEL_EXT))
    return candidates[0] if candidates else None

def _load_report() -> dict:
    """Load the YAML report from the standard report path."""
    if not REPORT_FILE.exists():
        raise FileNotFoundError(f"Report not found: {REPORT_FILE}")
    with open(REPORT_FILE) as f:
        return yaml.safe_load(f)

def _save_report(report: dict) -> None:
    """Persist *report* to the standard report path."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    print(f"[esp_monitor_parser] Report written to {REPORT_FILE}")


# monitor report functions

def parse_monitor_output(lines: list[str]) -> dict:
    """
    Parse ESP32 serial monitor output and write a YAML report to
    ./output/04_reports/monitor_report.yaml.
    (File will be moved to a framework-specific subdir by finalize_monitor_report() later.)

    Extracts:
      - setup time (µs)         — block 1: GetFeatureConfig, AllocateTensors, …
      - preprocessing time (µs) — block 2: FFT, Mel filterbank, …
      - inference time (µs)     — block 3: CONV, DEPTHWISE_CONV, FULLY_CONNECTED, …
      - total time (µs)         — sum of the three blocks above
      - RAM arena allocation    — total, head, tail
      - CRC32 checksums         — audio_input, features_output, model_input, model_output

    Model size and MACs are intentionally left as None here; call
    finalize_monitor_report() after the pipeline has finished to fill them in.
    """
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "framework": None,
        "model_size_bytes": None,
        "estimated_macs": None,
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

    timing_block_index = 0
    in_timing_block    = False

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
                    report["timing_us"]["setup"] = us
                elif timing_block_index == 2:
                    report["timing_us"]["preprocessing"] = us
                elif timing_block_index == 3:
                    report["timing_us"]["inference"] = us
                in_timing_block = False
                continue

        # CRC32 checksums
        m = re.match(r"Audio Input CRC32:\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            report["crc32"]["audio_input"] = m.group(1)
            continue

        m = re.match(r"Output Features CRC32:\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            report["crc32"]["features_output"] = m.group(1)
            continue

        m = re.match(r"Input CRC32:\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            report["crc32"]["model_input"] = m.group(1)
            continue

        m = re.match(r"Output CRC32:\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            report["crc32"]["model_output"] = m.group(1)
            continue

        # RAM / arena allocation
        m = re.match(r"\[RecordingMicroAllocator\] Arena allocation total\s+(\d+)\s+bytes", line)
        if m:
            report["ram_bytes"]["arena_total"] = int(m.group(1))
            continue

        m = re.match(r"\[RecordingMicroAllocator\] Arena allocation head\s+(\d+)\s+bytes", line)
        if m:
            report["ram_bytes"]["arena_head"] = int(m.group(1))
            continue

        m = re.match(r"\[RecordingMicroAllocator\] Arena allocation tail\s+(\d+)\s+bytes", line)
        if m:
            report["ram_bytes"]["arena_tail"] = int(m.group(1))
            continue

    # Derived total time
    times = [report["timing_us"][k] for k in ("setup", "preprocessing", "inference")]
    if all(t is not None for t in times):
        report["timing_us"]["total"] = sum(times)

    _save_report(report)
    return report

def finalize_monitor_report(framework: str) -> dict:
    """
    Add information about framework, model size, and MACs to existing YAML.

    Parameters
    ----------
    framework : {"tensorflow", "pytorch"}
        Used framework determines the used subfolder for model lookup and report output:
          - "tensorflow" → ./output/03_models/tensorflow/ & ./output/04_reports/tensorflow/
          - "pytorch"    → ./output/03_models/pytorch/ & ./output/04_reports/pytorch/

    Returns
    -------
    dict
        The updated report (also persisted to disk).
    """
    if framework not in FRAMEWORK_MODEL_DIRS:
        raise ValueError(
            f"Unknown framework {framework!r}. "
            f"Must be one of: {sorted(FRAMEWORK_MODEL_DIRS)}"
        )

    report = _load_report()
    report["framework"] = framework

    model_dir = FRAMEWORK_MODEL_DIRS[framework]
    tflite    = _find_tflite_in(model_dir)

    if tflite is None:
        print(
            f"[esp_monitor_parser] Warning: no {TFLITE_MODEL_EXT} model found "
            f"in {model_dir}"
        )
        report["model_size_bytes"] = None
        report["estimated_macs"]   = None
    else:
        report["model_size_bytes"] = tflite.stat().st_size
        try:
            report["estimated_macs"] = compute_macs(tflite)
        except Exception as e:
            print(f"[esp_monitor_parser] Error computing MACs: {e}")
            report["estimated_macs"] = None


    # Write into the framework subfolder and remove the staging file
    final_dir  = FRAMEWORK_REPORT_DIRS[framework]
    final_path = final_dir / "monitor_report.yaml"
    final_dir.mkdir(parents=True, exist_ok=True)
 
    with open(final_path, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    print(f"[esp_monitor_parser] Report written to {final_path}")
 
    REPORT_FILE.unlink(missing_ok=True)
 
    return report