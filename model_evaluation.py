import re
import yaml
import numpy as np
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score

# TODO fix data methods

def top1_accuracy_sklearn(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    pred_classes = np.argmax(y_pred, axis=1)

    return accuracy_score(y_true, pred_classes)


def filter_files_with_config(files, cfg_filter_files={'is_used': False, 're_contains': '.*'}):
    """
    filter files with config {filter_files: {is_used, re_contains}}
    """

    # returns
    if cfg_filter_files is None: return files
    if not cfg_filter_files['is_used']: return files

    # assert string in re
    assert isinstance(cfg_filter_files['re_contains'], str)

    # filter files
    filtered_files = [f for f in files if re.search(cfg_filter_files['re_contains'], str(f))]

    # assert that there are still some files left
    assert len(filtered_files), 'No files left after filtering!!! Change in config.yaml -> filter_files.re_contains'

    # filter files
    return filtered_files

def loadCalibrationDataset(cfg, key="train", cache_id=None):
    """
    load data from cache for calibration
    """

    additional_file_filter_cfg={'is_used': True, 're_contains': key}

    # target cached path
    target_cached_path = Path(cfg['caching']['root_path']) / cfg['caching']['cache_id']

    # target cache id 
    if not cache_id is None:

      # change cached path
      cached_path=Path(cfg['caching']['root_path']) / cfg['caching']['cache_id']
      target_cached_path = cached_path.parent / cache_id

      # check if path exists
      if not target_cached_path.exists(): raise ValueError('cach_id: {} does not exist in path: {}!'.format(cache_id, cached_path.parent))

    # cached files
    cached_files_filtered = filter_files_with_config(sorted(list(target_cached_path.glob('**/*.npz'))), cfg['load_cache']['filter_files'])

    # additional filtering
    cached_files_filtered = filter_files_with_config(cached_files_filtered, additional_file_filter_cfg)


    cache_info=yaml.safe_load(open(str(target_cached_path / 'cache_info.yaml')))

    calibration_samples = []
    labels = []

    for i, cached_file in enumerate(cached_files_filtered):
      # data 
      data = np.load(cached_file)
      x = data['x'].reshape(cache_info['feature_size_origin'])
      labels.append(data['y'])

      if not cfg['feature_handler_add_kwargs']['add_channel_dimension']: x = x[np.newaxis, :]

      calibration_samples.append(x)
      data.close()

    return np.array(calibration_samples), np.array(labels)

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
    sample = sample.reshape(1, *sample.shape)  # tflite still needs batch dim
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    if quantized:
      y_pred = (y_pred.astype(np.float32) - output_zero_point) * output_scale
    y_preds.append(y_pred[0])

  return np.array(y_preds)


def model_evaluation(cfg, tflitepath):
  print('Running inference from model save in ', tflitepath)
  testset, y_true =loadCalibrationDataset(cfg, key='test')
  
  y_pred= run_model(testset, tflitepath)
  y_prob = softmax(y_pred, axis=1)

  auc = roc_auc_score(y_true, y_prob,
            multi_class="ovr",   # important for multiclass
            average="macro")
  
  print('Top-1 accuracy: ', top1_accuracy_sklearn(y_pred, y_true))
  print('Area under ROC curve: ', auc)

  print('Completed evaluations for model save in ', tflitepath)
   

if __name__ == '__main__':
    # yaml config file
  cfg = yaml.safe_load(open('./config.yaml'))['datamodule']

  testset, y_true =loadCalibrationDataset(cfg, key='test')
  
  y_pred= run_model(testset, "output/03_model/ModelTinyMl.tflite")
  print(top1_accuracy_sklearn(y_pred, y_true))
  y_pred = run_model(testset, "output/03_model/ModelTinyMlINT8.tflite")
  print(top1_accuracy_sklearn(y_pred, y_true))

  from scipy.special import softmax

  y_prob = softmax(y_pred, axis=1)

  auc = roc_auc_score(y_true, y_prob,
            multi_class="ovr",   # important for multiclass
            average="macro")
  print(auc)

  # Top-1 accuracy
