# --
# paths

from pathlib import Path


OUT_DIR = Path(__file__).parent.parent / 'output'

PARENT_MODELS_DIR =  OUT_DIR / '03_models'
MODELS_DIR = PARENT_MODELS_DIR / 'tensorflow'
KERAS_MODEL_PATH = MODELS_DIR / 'model.keras'
TFLITE_MODEL_PATH = MODELS_DIR / 'model.tflite'
REFERENCE_DATASET_PATH = MODELS_DIR / "reference_dataset"


PARENT_REPORTING_DIR = OUT_DIR / '04_reports' 
REPORTING_DIR = PARENT_REPORTING_DIR / 'tensorflow'
TENSORBOARD_LOGS_PATH = REPORTING_DIR / "tensorboard"
CM_FIG_PATH = REPORTING_DIR / "cm.png"

# create directories
[p.mkdir() for p in [OUT_DIR, PARENT_MODELS_DIR, MODELS_DIR, PARENT_REPORTING_DIR, REPORTING_DIR] if not p.exists()]