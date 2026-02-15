# --
# biodcase - main pipeline

from config import load_config
from data_preprocessing import run_data_preprocessing
from feature_extraction import run_feature_extraction
from model_training import run_model_training
from embedded_code_generation import run_embedded_code_generation
from deploy_generated_code import run_deploy_generated_code
from paths import KERAS_MODEL_PATH, REFERENCE_DATASET_PATH, TFLITE_MODEL_PATH, GEN_CODE_DIR


if __name__ == '__main__':
  """
  biodcase - main pipeline
  """

  # config
  config = load_config()

  # preprocessing
  run_data_preprocessing(config)

  # feature extractions
  run_feature_extraction(config)

  # model training
  run_model_training(config)

  # embedded code generation
  run_embedded_code_generation(config, model_path=KERAS_MODEL_PATH, reference_dataset_path=REFERENCE_DATASET_PATH, tflite_model_path=TFLITE_MODEL_PATH, gen_code_dir=GEN_CODE_DIR)

  # deployment
  run_deploy_generated_code(config)
