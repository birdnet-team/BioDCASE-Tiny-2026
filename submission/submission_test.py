# --
# submission test

import sys
import yaml
import numpy as np
import soundfile

from pathlib import Path

# required package paths - root path
[sys.path.append(p) for p in[str(Path(__file__).parent.parent)] if p not in sys.path]

from plots import plot_confusion_matrix
from embedded_code_generation import run_compile_embedded_src_code, run_create_target_embedded_src_code, run_deploy_embedded_compiled_code
from inference_handler import InferenceHandler
from biodcase_tiny.embedded.esp_monitor_parser import compute_macs


def run_inference(cfg, inference_scores_file, cm_plot_file):
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
  y_predictions_model, y_predictions_tflite = [], []
  inference_score_dict = {}

  # run through each test file
  for test_file in test_files:

    # target
    y_target = inference_handler.get_label_dict()[test_file.parent.stem]

    # read audio
    waveform, fs = soundfile.read(test_file)

    # infer
    y_hat_model, y_hat_tflite = inference_handler.infer(waveform, fs)

    # add target and prediction
    y_targets.append(y_target)
    y_predictions_model.extend(y_hat_model)
    if y_hat_tflite is not None: y_predictions_tflite.extend(y_hat_tflite)

  # to numpy
  y_targets = np.array(y_targets)
  y_predictions_model = np.array(y_predictions_model)
  y_predictions_tflite = np.array(y_predictions_tflite)

  # add to score dict
  inference_score_dict['accuracy_model'] = np.round(np.mean(y_targets == np.argmax(y_predictions_model, axis=-1)), decimals=4).item()
  inference_score_dict['accuracy_tflite'] = np.round(np.mean(y_targets == np.argmax(y_predictions_tflite, axis=-1)), decimals=4).item()

  # todo:
  inference_score_dict['accuracy'] = np.round(np.mean(y_targets == np.argmax(y_predictions_model, axis=-1)), decimals=4).item()
  inference_score_dict['inference_model_size'] = inference_handler.get_model_size()

  # tflite file
  target_tflite_file = inference_handler.get_tflite_model_file()

  # tflite model size
  inference_score_dict['tflite_model_size'] = target_tflite_file.stat().st_size if target_tflite_file.is_file() else 'N/A'

  # info message
  if not target_tflite_file.is_file(): print("***No tflite model, name must be same as the inference model but with .tflite as ending, e.g.: {}".format(target_tflite_file))

  # info score
  print("\nInference results:")
  print("y_targets: ", y_targets)
  print("y_predictions_model argmax: ", np.argmax(y_predictions_model, axis=-1))
  print("Accuracy: {:.4f}".format(inference_score_dict['accuracy']))

  # todo:
  # add macs / num params 

  # dump scores
  yaml.dump({'inference_score_dict': inference_score_dict}, open(inference_scores_file, 'w'))

  # plot
  plot_confusion_matrix(y_targets, np.argmax(y_predictions_model, axis=-1), labels=inference_handler.get_label_dict().keys(), plot_path=cm_plot_file, show_plot_flag=False)

  return target_tflite_file


def run_embedded(cfg, tflite_path, monitor_report_file):
  """
  run embedded
  """

  # src path
  src_path = Path(cfg['generate_embedded_code']['gen_code_dir']) / cfg['generate_embedded_code']['gen_code_source_folder_name']

  # gen code available just deploy it - skip building
  if src_path.is_dir():

    # code
    try: 
      run_deploy_embedded_compiled_code(cfg)
    except:
      print("\n***Could not deploy your compiled code!")

  # no gen code available - build it first (only works if .tflite model is saved)
  else:

    # tflite not available - skip
    if tflite_path is None: return
        
    # info
    print(".tflite available: {}\ncreate, build, deploy...".format(tflite_path))

    # todo: solve this in a better way
    # add feature params - for target creation
    cfg['datamodule'] = {}
    cfg['datamodule']['feature_extraction'] = cfg['inference_handler']['feature_handler']['kwargs']
    cfg['datamodule']['target_sample_rate'] = cfg['inference_handler']['target_sample_rate']

    # run generate embedded src code
    run_create_target_embedded_src_code(cfg, tflite_path)

    # compile
    run_compile_embedded_src_code(cfg)

    # deploy
    try:
      run_deploy_embedded_compiled_code(cfg)
    except:
      print("\n***Could not deploy compiled code!")

    # todo:
    # macs
    macs = compute_macs(tflite_path)
    print("macs: ", macs)

  # todo:
  # copy monitor report file to report dir location!
  from biodcase_tiny.embedded.esp_monitor_parser import REPORT_FILE as _monitor_report_file
    
  # does not exist - skip
  if not _monitor_report_file.is_file(): return

  # copy file
  yaml.dump(yaml.safe_load(open(_monitor_report_file, 'r')), open(monitor_report_file, 'w'), default_flow_style=False, sort_keys=False)


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
  cfg = yaml.safe_load(open('config_submission.yaml'))

  # report dir
  #report_dir = Path(__file__).parent / 'reports'
  report_dir = Path(cfg['report_dir'])

  # create directory
  if not report_dir.is_dir(): report_dir.mkdir()

  # files
  submission_results_file = report_dir / 'submission_results.yaml'
  inference_scores_file = report_dir / 'inference_scores.yaml'
  monitor_report_file = report_dir / 'monitor_report.yaml'
  cm_plot_file = report_dir / 'cm_inference.png'

  # remove previous generated files
  [(print("remove: ", f.name), f.unlink()) for f in [submission_results_file, inference_scores_file, monitor_report_file, cm_plot_file] if f.is_file()]

  # inference
  tflite_path = run_inference(cfg, inference_scores_file, cm_plot_file)

  # embedded
  run_embedded(cfg, tflite_path, monitor_report_file)

  # write final results
  submission_result_dict = run_write_final_results(cfg, inference_scores_file, monitor_report_file, submission_results_file)

  # until now everything run successfully -> save total scores 
  print("\nSuccessful submission test run!")
  #print("Your results:\n{}".format(submission_result_dict))
  print("Your results:")
  [print("{}: {}".format(k, v)) for k, v in submission_result_dict.items()]