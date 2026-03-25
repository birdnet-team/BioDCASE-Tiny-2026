import re
import yaml
from pathlib import Path
from datetime import datetime

# Model location config
MODEL_DIRS  = [Path("./output/03_model"), Path("./output/03_models")]
MODEL_NAMES = ["model.tflite", "ModelTinyMl.tflite"]


def _find_tflite() -> Path | None:
    """Return the first matching .tflite file across all known dirs and names."""
    for folder in MODEL_DIRS:
        for name in MODEL_NAMES:
            candidate = folder / name
            if candidate.exists():
                return candidate
    return None


def parse_monitor_output(lines: list[str], report_dir: Path = Path(".")) -> dict:
    """
    Parse ESP32 serial monitor output and write a YAML report.

    Extracts:
      - model size (bytes) from .tflite file found via MODEL_DIRS / MODEL_NAMES
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
    else:
        print(f"[esp_monitor_parser] Warning: no .tflite found in {MODEL_DIRS} with names {MODEL_NAMES}")

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
    report_path = Path(report_dir) / "monitor_report.yaml"
    with open(report_path, "w") as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)

    print(f"[esp_monitor_parser] Report written to {report_path}")
    return result