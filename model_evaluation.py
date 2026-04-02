import re
import yaml
import numpy as np
from pathlib import Path

from datamodule import DatamoduleTinyMl
from ai_edge_litert.interpreter import Interpreter

from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score

# TODO fix data methods

def top1_accuracy_sklearn(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    pred_classes = np.argmax(y_pred, axis=1)

    return accuracy_score(y_true, pred_classes)

def run_model(input_data, model_path):
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


def model_evaluation(cfg_datamodule, tflitepath):
  datamodule_calibrate = DatamoduleTinyMl(cfg_datamodule, load_set_on_init='test')
  X_test, y_true = datamodule_calibrate.features, datamodule_calibrate.targets
  
  y_pred= run_model(X_test, tflitepath)
  y_prob = softmax(y_pred, axis=1)

  auc = roc_auc_score(y_true, y_prob,
            multi_class="ovr",   # important for multiclass
            average="macro")
  
  print('Top-1 accuracy: ', top1_accuracy_sklearn(y_pred, y_true))
  print('Area under ROC curve: ', auc)

  print('Completed evaluations for model save in ', tflitepath)
   

if __name__ == '__main__':
  # yaml config file

  cfg_datamodule = yaml.safe_load(open('./config.yaml'))['datamodule']

  model_evaluation(cfg_datamodule, "output/03_models/Baseline.tflite")
  model_evaluation(cfg_datamodule, "output/03_models/Baseline.tflite")