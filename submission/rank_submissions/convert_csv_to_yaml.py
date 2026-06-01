# Reads the evaluation results CSV produced by evaluate_submissions.py and converts
# it into a structured YAML file for the leaderboard website.
# Only contestants without errors are included.
# rank_team and rank_score are set to N/A and filled in later.
#
# Output file: results_task3_entries.yaml (same folder as the input CSV / this script)

import csv
from pathlib import Path


def _format_value(val):
    """
    Convert a CSV string value to the most appropriate scalar type for YAML output.
    Preserves N/A as an unquoted string. Tries int, then float, then falls back to string.
    """
    if val is None or str(val).strip() in ("N/A", ""):
        return "N/A"
    val = str(val).strip()
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _entry_to_yaml_block(contestant, row, is_first):
    """
    Render one contestant entry as a YAML block string, matching the required layout
    including section comments and blank lines between sections.
    """
    # derive a clean name and bibtex key from the contestant folder name
    # e.g. "Mustermann_task3_1" -> name: "Mustermann_task3_1", bibtex_key: "mustermann2026"
    name = contestant
    bibtex_key = contestant.split("_task3")[0].lower() + "2026"

    v = {field: _format_value(row.get(field)) for field in [
        "accuracy_inference", "roc_auc_inference", "model_size_inference_bytes",
        "macs_inference", "num_params_inference",
        "accuracy_tflite", "roc_auc_tflite", "model_size_tflite_bytes",
        "macs_tflite", "num_params_tflite",
        "embedded_time_ms_setup", "embedded_time_ms_preprocessing",
        "embedded_time_ms_model", "embedded_time_ms_total", "embedded_ram_usage_bytes",
    ]}

    # separator comment before every entry except the first
    separator = "" if is_first else "\n\n# next entry\n"

    return (
        f"{separator}"
        f"- hline: false\n"
        f"\n"
        f"  # submission information\n"
        f"  name: {name}\n"
        f"  bibtex_key: {bibtex_key}\n"
        f"\n"
        f"  # rank\n"
        f"  rank_team: N/A\n"
        f"  rank_score: N/A\n"
        f"\n"
        f"  # inference model\n"
        f"  accuracy_inference: {v['accuracy_inference']}\n"
        f"  roc_auc_inference: {v['roc_auc_inference']}\n"
        f"  model_size_inference_bytes: {v['model_size_inference_bytes']}\n"
        f"  macs_inference: {v['macs_inference']}\n"
        f"  num_params_inference: {v['num_params_inference']}\n"
        f"\n"
        f"  # embedding model\n"
        f"  accuracy_tflite: {v['accuracy_tflite']}\n"
        f"  roc_auc_tflite: {v['roc_auc_tflite']}\n"
        f"  model_size_tflite_bytes: {v['model_size_tflite_bytes']}\n"
        f"  macs_tflite: {v['macs_tflite']}\n"
        f"  num_params_tflite: {v['num_params_tflite']}\n"
        f"\n"
        f"  # embedded performance on esp32-s3\n"
        f"  embedded_time_ms_setup: {v['embedded_time_ms_setup']}\n"
        f"  embedded_time_ms_preprocessing: {v['embedded_time_ms_preprocessing']}\n"
        f"  embedded_time_ms_model: {v['embedded_time_ms_model']}\n"
        f"  embedded_time_ms_total: {v['embedded_time_ms_total']}\n"
        f"  embedded_ram_usage_bytes: {v['embedded_ram_usage_bytes']}\n"
    )


def convert_csv_to_yaml(csv_path):
    """
    Read the evaluation results CSV, filter out error rows, and write a structured
    YAML file (results_task3_entries.yaml) into the same folder as the CSV.
    """
    csv_path = Path(csv_path)
    output_path = csv_path.parent / "results_task3_entries.yaml"

    # read all rows from the csv
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # only keep rows that completed without an error
    valid_rows = [r for r in rows if not r.get("error", "").strip()]

    if not valid_rows:
        print("convert_csv_to_yaml: no valid (error-free) rows found, skipping yaml output.")
        return

    # build the full yaml string entry by entry
    blocks = []
    for idx, row in enumerate(valid_rows):
        contestant = row.get("Contestant", f"unknown_{idx}")
        blocks.append(_entry_to_yaml_block(contestant, row, is_first=(idx == 0)))

    yaml_text = "".join(blocks)

    output_path.write_text(yaml_text, encoding="utf-8")
    print(f"convert_csv_to_yaml: wrote {len(valid_rows)} entr{'y' if len(valid_rows) == 1 else 'ies'} to '{output_path}'.")


if __name__ == "__main__":
    # when run directly, expect the csv path as a command-line argument
    import sys
    if len(sys.argv) != 2:
        print("Usage: python convert_csv_to_yaml.py <path_to_eval_results.csv>")
        sys.exit(1)
    convert_csv_to_yaml(sys.argv[1])