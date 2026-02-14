from pathlib import Path

PIPELINE_CONFIG_FILE = Path(__file__).parent / "pipeline_config.yaml"

OUT_DIR = Path(__file__).parent / 'your_output'

RAW_OUT_DIR = OUT_DIR / "00_raw"

# change this to your dataset path
CLIPS_DIR = Path("/world/dekutree/datasets/biodcase2026_task3_tiny_ml/raw")
#CLIPS_DIR = RAW_OUT_DIR / "clips"

PREPROC_DIR = OUT_DIR / "01_intermediate"
PREPROC_PRQ_PATH = PREPROC_DIR / "preproc.parquet"

FEATURES_DIR = OUT_DIR / "02_features"
FEATURES_PRQ_PATH = FEATURES_DIR / "features.parquet"
FEATURES_SHAPE_JSON_PATH = FEATURES_DIR / "features_shape.json"
FEATURES_SAMPLE_PLOT_PATH = FEATURES_DIR / "features_sample.png"
FEATURES_SAMPLE_PLOT_DIR = FEATURES_DIR / "sample_plots"

MODELS_DIR = OUT_DIR / '03_models'
KERAS_MODEL_PATH = MODELS_DIR / 'model.keras'
TFLITE_MODEL_PATH = MODELS_DIR / 'model.tflite'
REFERENCE_DATASET_PATH = MODELS_DIR / "reference_dataset"

REPORTING_DIR = OUT_DIR / '04_reporting'
TENSORBOARD_LOGS_PATH = REPORTING_DIR / "tensorboard"
CM_FIG_PATH = REPORTING_DIR / "cm.png"

GEN_CODE_DIR = OUT_DIR / '05_generated_code'

EVAL_ZIP_PATH = RAW_OUT_DIR / "Evaluation Set.zip"
EVAL_CLIPS_DIR = RAW_OUT_DIR / "eval_clips"
