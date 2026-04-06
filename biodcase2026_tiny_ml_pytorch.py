# --
# biodcase 2026 tiny ml - main pipeline pytorch

import sys
import yaml
from pathlib import Path

from datamodule import DatamoduleTinyMl
from pipeline_pytorch.model_training import pytorch_model_taining
from pipeline_pytorch.paths import MODELS_DIR
from embedded_code_generation import run_compile_embedded_src_code, run_create_target_embedded_src_code, run_deploy_embedded_compiled_code
from model_evaluation import model_evaluation
from model_quantization import model_quantization
from biodcase_tiny.embedded.esp_monitor_parser import finalize_monitor_report

if __name__ == '__main__':
  """
  tiny ml starts here
  """

  # yaml config file
  cfg = yaml.safe_load(open('./config.yaml'))

  # info
  print("Hello Tiny ML 2026, version: {}".format(cfg['version']))

  datamodule_train = DatamoduleTinyMl(cfg['datamodule'], load_set_on_init='train')
  datamodule_validation = DatamoduleTinyMl(cfg['datamodule'], load_set_on_init='validation')
  datamodule_test = DatamoduleTinyMl(cfg['datamodule'], load_set_on_init='test')
  datamodule_train.info()

  # run model training and test in pytorch
  pytorch_model_taining(cfg, datamodule_train, datamodule_validation, datamodule_test)


  # Quantization and evaluation
  submission_cfg=yaml.safe_load(open('./submission/config.yaml'))
  model_name=submission_cfg['inference_handler_pytorch']['model']['attr']
  tflite_path = MODELS_DIR / f"{model_name}.tflite"

  print(tflite_path)

  # quantize
  if cfg['generate_embedded_code']['quantize']:
    print("Model evaluation before quantization: ")
    model_evaluation(cfg['datamodule'], tflite_path)

    # TODO fix overwritten file (add quantization path)
    print("Model quantization (model will be overwritten!) ")
    model_quantization(cfg['datamodule'], tflite_path, tflite_path)

    print("Model evaluation after quantization: ")
  
  # evaluation .tflite model  
  model_evaluation(cfg['datamodule'], tflite_path)

  # skip deployment?
  if cfg['skip_deployment_flag']:
    print("\nSkip deployment! For deployment change 'skip_deployment_flag' to 'False' in 'config.yaml'")
    sys.exit()

  # run generate embedded src code
  run_create_target_embedded_src_code(cfg, tflite_path)

  # compile
  run_compile_embedded_src_code(cfg)

  # deploy
  run_deploy_embedded_compiled_code(cfg)

  # finalize the monitor report yaml
  finalize_monitor_report("pytorch")