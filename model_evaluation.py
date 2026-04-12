# --
# model evaluation

import re
import yaml
import numpy as np
import importlib

from pathlib import Path
from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score

from datamodule import DatamoduleTinyMl
from ai_edge_litert.interpreter import Interpreter


def top1_accuracy_sklearn(y_pred, y_true):
  """
  top1 accuracy
  """

  # to numpy
  y_pred = np.array(y_pred)
  y_true = np.array(y_true)

  # argmax
  pred_classes = np.argmax(y_pred, axis=1)

  # score
  return accuracy_score(y_true, pred_classes)


def run_model_main(cfg, datamodule_test, model_path):
  """
  run model depending on suffix
  """

  # make sure its a path
  model_path = Path(model_path)

  # start
  if model_path.suffix == '.pth': return run_model_pytorch(cfg, datamodule_test, model_path)
  if model_path.suffix == '.keras': return run_model_tensorflow(cfg, datamodule_test, model_path)
  if model_path.suffix == '.tflite': return run_model_tflite(cfg, datamodule_test, model_path)

  # could not find suffix
  print("***Model evaluation at path: {} could not be executed due to not specified suffix: {}".format(model_path, model_path.suffix))

  return None


def run_model_pytorch(cfg, datamodule_test, model_path):
  """
  run pytorch model
  """

  # model class
  model_class = getattr(importlib.import_module(cfg['pytorch_framework']['model']['module']), cfg['pytorch_framework']['model']['attr'])

  # model kwargs
  model_kwargs_overwrite = {'input_shape': datamodule_test.get_feature_shape_at_load(), 'num_classes': len(datamodule_test.get_label_dict()), 'is_inference_model': True}

  # model
  model = model_class(*cfg['pytorch_framework']['model']['args'], **{**cfg['pytorch_framework']['model']['kwargs'], **model_kwargs_overwrite})

  # load model weights
  model.load(model_path)

  # do prediction
  return model.predict(datamodule_test.features)


def run_model_tensorflow(cfg, datamodule_test, model_path):
  """
  run tensorflow model
  """
  import keras

  # model instance
  model = keras.models.load_model(model_path)

  # channel dim at end
  features = np.transpose(datamodule_test.features, (0, 2, 3, 1))

  # do prediction
  return model.predict(features)


def run_model_tflite(cfg, datamodule_test, model_path):
  """
  run tflite model
  """

  # input data
  input_data = datamodule_test.features

  # tflite interpreter
  interpreter = Interpreter(model_path=str(model_path))
  interpreter.allocate_tensors()

  # details
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # types
  input_dtype = input_details[0]["dtype"]
  output_dtype = output_details[0]["dtype"]

  # type setup
  if input_dtype == np.int8 and output_dtype == np.int8:
    input_scale = input_details[0]['quantization_parameters']['scales'][0]
    input_zero_point = input_details[0]['quantization_parameters']['zero_points'][0]
    output_scale = output_details[0]['quantization_parameters']['scales'][0]
    output_zero_point = output_details[0]['quantization_parameters']['zero_points'][0]
    input_data = np.clip(input_data / input_scale + input_zero_point, -128, 127).astype(input_dtype)
    quantized = True
  elif input_dtype == np.float32 and output_dtype == np.float32:
    quantized = False
  else:
    raise NotImplementedError(
      f"Case where input dtype is {input_dtype} "
      f"and output dtype is {output_dtype} not supported"
    )

  # predictions
  y_preds = []

  # run samples
  for sample in input_data:
    sample = sample.reshape(input_details[0]['shape'])
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    if quantized:
      y_pred = (y_pred.astype(np.float32) - output_zero_point) * output_scale
    y_preds.append(y_pred[0])

  return np.array(y_preds)


def model_evaluation(cfg, datamodule_test, model_path):
  """
  model evaluation
  """
  
  # get features and targets
  #X_test, y_true = datamodule_test.features, datamodule_test.targets
  y_true = datamodule_test.targets
  
  # run model
  y_pred = run_model_main(cfg, datamodule_test, model_path)

  # skip -> model could not be executed
  if y_pred is None: return

  # softmax
  y_prob = softmax(y_pred, axis=1)

  # auc score
  auc = roc_auc_score(y_true, y_prob,
            multi_class="ovr",   # important for multiclass
            average="macro")
  
  # prints
  print('Top-1 accuracy: {:.4f}'.format(top1_accuracy_sklearn(y_pred, y_true)))
  print('Area under ROC curve: {:.4f}'.format(auc))
  print('Completed evaluations for model saved in ', model_path)
   

if __name__ == '__main__':
  """
  model evaluation
  """

  # yaml config file
  cfg = yaml.safe_load(open('./config.yaml'))

  # test datamodule
  datamodule_test = DatamoduleTinyMl(cfg['datamodule'], load_set_on_init='test')

  # models dir
  from pipeline_pytorch.paths import MODELS_DIR as pytorch_model_dir
  from pipeline_tensorflow.paths import MODELS_DIR as tensorflow_model_dir

  # supported suffixes
  supported_suffixes = ['.tflite', '.pth', '.keras']

  # get all 
  f_find_all_model_files = lambda model_dir, supported_suffixes: [model_file for model_files in [list(model_dir.glob('*{}'.format(s))) for s in supported_suffixes] for model_file in model_files]

  # run evaluation
  for model_dir in [pytorch_model_dir, tensorflow_model_dir]:

    # for each model in model dir
    for model_path in f_find_all_model_files(model_dir, supported_suffixes):

      # info
      print("\nEvaluate: ", model_path)

      # run evaluation
      model_evaluation(cfg, datamodule_test, model_path)
