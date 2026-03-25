# --
# submission test

import sys
import yaml
import numpy as np
import soundfile

from pathlib import Path
from inference_handler import InferenceHandler

# required package paths
[sys.path.append(p) for p in[str(Path(__file__).parent.parent)] if p not in sys.path]

from plots import plot_confusion_matrix


if __name__ == '__main__':
  """
  submission test
  """

  # yaml config file
  cfg = yaml.safe_load(open('config_inference.yaml'))

  # report dir
  report_dir = Path(__file__).parent / 'reports'

  # assertion
  assert report_dir.exists(), "Something is wrong with your report directory!"

  # inference handler
  inference_handler = InferenceHandler(cfg['inference_handler'])

  # path to test files
  test_files = sorted(list(Path(cfg['test_file_dir']).glob('**/*' + cfg['test_files_ext'])))

  # collect targets and predictions
  y_targets = []
  y_predictions = []
  score_dict = {}

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

  # info
  print("\nSubmission test:")
  print("y_targets: ", y_targets)
  print("y_predictions: ", y_predictions)

  # accuracy
  acc = np.round(np.mean(y_targets == y_predictions), decimals=6).item()

  # add to score dict
  score_dict['accuracy'] = acc

  # info score
  print("accuracy: {:.4f}".format(acc))
  print("\nTest submission successfuly run!")

  # dump scores
  yaml.dump({'score_dict': score_dict}, open(report_dir / 'score_dict.yaml', 'w'))

  # plot
  plot_confusion_matrix(y_targets, y_predictions, labels=inference_handler.get_label_dict().keys(), plot_path=report_dir / 'test_cm.png', show_plot_flag=True)
