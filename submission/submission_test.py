# --
# submission test

import sys
import yaml
import numpy as np
import soundfile

from pathlib import Path
from inference_handler import InferenceHandler

# required package paths - root path
[sys.path.append(p) for p in[str(Path(__file__).parent.parent)] if p not in sys.path]

from plots import plot_confusion_matrix
from biodcase2026_tiny_ml_pytorch import run_deploy_embedded_compiled_code


def run_inference(cfg, inference_scores_file):
  """
  run inference
  """

  # inference handler
  inference_handler = InferenceHandler(cfg['inference_handler'])

  # for testing
  #inference_handler = InferenceHandler(cfg['inference_handler_tensorflow'])
  #inference_handler = InferenceHandler(cfg['inference_handler_pytorch'])

  # info
  inference_handler.info()

  # path to test files
  test_files = sorted(list(Path(cfg['test_file_dir']).glob('**/*' + cfg['test_files_ext'])))

  # collect targets and predictions
  y_targets = []
  y_predictions = []
  inference_score_dict = {}

  # run through each test file
  for test_file in test_files:

    # target
    y_target = inference_handler.get_label_dict()[test_file.parent.stem]

    # read audio
    waveform, fs = soundfile.read(test_file)

    # infer
    y_hat = inference_handler.infer(waveform, fs)

    # add target and prediction
    y_targets.append(y_target)
    y_predictions.extend(y_hat)

  # to numpy
  y_targets = np.array(y_targets)
  y_predictions = np.array(y_predictions)

  # accuracy
  acc = np.round(np.mean(y_targets == y_predictions), decimals=6).item()

  # add to score dict
  inference_score_dict['accuracy'] = acc
  inference_score_dict['inference_model_size'] = inference_handler.get_model_size()

  print(inference_handler.get_model_file())
  target_tflite_file = Path(inference_handler.get_model_file()).parent / (Path(inference_handler.get_model_file()).stem + '.tflite')

  # tflite model size
  inference_score_dict['tflite_model_size'] = target_tflite_file.stat().st_size if target_tflite_file.is_file() else 'N/A'

  # info message
  if not target_tflite_file.is_file(): print("***No tflite model, name must be same as the inference model but with .tflite as ending, e.g.: {}".format(target_tflite_file))

  # info score
  print("\nInference results:")
  print("y_targets: ", y_targets)
  print("y_predictions: ", y_predictions)
  print("Accuracy: {:.4f}".format(acc))
  print("\nSuccessful submission test run!")

  # dump scores
  yaml.dump({'inference_score_dict': inference_score_dict}, open(inference_scores_file, 'w'))

  # plot
  plot_confusion_matrix(y_targets, y_predictions, labels=inference_handler.get_label_dict().keys(), plot_path=report_dir / 'cm_inference.png', show_plot_flag=False)


def run_write_final_results(cfg, inference_scores_file, monitor_report_file, submission_results_file):
  """
  write final results
  """

  # your submission results init -> will be overwritten
  submission_result_dict = {
    'average_precision': 0.0,
    'inference_model_size_bytes': 'N/A',
    'tflite_model_size_bytes': 'N/A',
    'ram_usage_bytes': 'N/A',
    'preprocessing_time_ms': 'N/A',
    'model_time_ms': 'N/A',
    'total_time_ms': 'N/A',
    }

  # scores
  if inference_scores_file.is_file():

    # inference scores
    inference_score_dict = yaml.safe_load(open(inference_scores_file, 'r'))['inference_score_dict']

    # change results
    submission_result_dict['average_precision'] = inference_score_dict['accuracy']
    submission_result_dict['inference_model_size_bytes'] = inference_score_dict['inference_model_size']
    submission_result_dict['tflite_model_size_bytes'] = inference_score_dict['tflite_model_size']

  # monitoring profiler info
  if monitor_report_file.is_file():

    # monitor report
    monitor_report_dict = yaml.safe_load(open(monitor_report_file, 'r'))

    # change
    #submission_result_dict['tflite_model_size_bytes'] = monitor_report_dict['model_size_bytes']
    submission_result_dict['preprocessing_time_ms'] = round(monitor_report_dict['timing_us']['preprocessing'] / 1000, 2)
    submission_result_dict['model_time_ms'] = round(monitor_report_dict['timing_us']['inference'] / 1000, 2)
    submission_result_dict['total_time_ms'] = round(monitor_report_dict['timing_us']['total'] / 1000, 2)
    submission_result_dict['ram_usage_bytes'] = monitor_report_dict['ram_bytes']['arena_total']

  # dump results
  yaml.dump({'submission_results': submission_result_dict}, open(submission_results_file, 'w'))
  print("Submission results written to: {}".format(submission_results_file))

  return submission_result_dict


if __name__ == '__main__':
  """
  submission test
  """

  # hide warnings of tensorflow
  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

  # yaml config file
  cfg = yaml.safe_load(open('config.yaml'))

  # report dir
  report_dir = Path(__file__).parent / 'reports'

  # create directory
  if not report_dir.is_dir(): report_dir.mkdir()

  # files
  submission_results_file = report_dir / 'submission_results.yaml'
  inference_scores_file = report_dir / 'inference_scores.yaml'
  monitor_report_file = report_dir / 'monitor_report.yaml'

  # remove previous generated files
  [(print("remove: ", f.name), f.unlink()) for f in [submission_results_file, inference_scores_file, monitor_report_file, *list(report_dir.glob('*.png'))] if f.is_file()]

  # inference
  run_inference(cfg, inference_scores_file)

  # code
  try: 
    # check if there is generated code
    run_deploy_embedded_compiled_code(cfg)

  except:
    print("\n***Could not deploy your compiled code!")

  # write final results
  submission_result_dict = run_write_final_results(cfg, inference_scores_file, monitor_report_file, submission_results_file)

  # until now everything run successfully -> save total scores 
  print("\nSuccessful submission test run!")
  print("Your results:\n{}".format(submission_result_dict))