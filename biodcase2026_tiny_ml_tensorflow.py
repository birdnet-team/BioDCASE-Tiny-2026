# --
# biodcase - main pipeline

import yaml
from pipeline_tensorflow.model_training import run_model_training
from pipeline_tensorflow.paths import TFLITE_MODEL_PATH

from datamodule import DatamoduleTinyMl
from embedded_code_generation import run_compile_embedded_src_code, run_create_target_embedded_src_code, run_deploy_embedded_compiled_code
from model_evaluation import model_evaluation
from model_quantization import model_quantization

if __name__ == '__main__':
  """
  biodcase - main pipeline
  """

  # yaml config file
  cfg = yaml.safe_load(open('./config.yaml'))

  # info
  print("Hello Tiny ML 2026, version: {}".format(cfg['version']))

  datamodule_train = DatamoduleTinyMl(cfg['datamodule'], load_set_on_init='train')
  datamodule_validation = DatamoduleTinyMl(cfg['datamodule'], load_set_on_init='validation')
  datamodule_test = DatamoduleTinyMl(cfg['datamodule'], load_set_on_init='test')
  datamodule_train.info()

  # model training
  run_model_training(cfg['tensorflow_training'], datamodule_train, datamodule_validation)


  # quantize
  if cfg['generate_embedded_code']['quantize']:
    print("Model evaluation before quantization: ")
    model_evaluation(cfg['datamodule'], TFLITE_MODEL_PATH)

    # TODO fix overwritten file (add quantization path)
    print("Model quantization (model will be overwritten!) ")
    model_quantization(cfg['datamodule'], TFLITE_MODEL_PATH, TFLITE_MODEL_PATH)

    print("Model evaluation after quantization: ")
  
  # evaluation .tflite model  
  model_evaluation(cfg['datamodule'], TFLITE_MODEL_PATH)

  # run generate embedded src code
  run_create_target_embedded_src_code(cfg, TFLITE_MODEL_PATH)

  # compile
  run_compile_embedded_src_code(cfg)

  # deploy
  run_deploy_embedded_compiled_code(cfg)
