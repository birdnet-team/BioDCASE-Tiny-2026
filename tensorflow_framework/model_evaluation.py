"""
***
Warning: 
This code was used from the previous biodcase challenge 2025, still not verified!!!
***

todo:
verify this code

This is the evaluation script for the BIODCASE competition.
To run it on evaluation data, place the "Evaluation Set.zip" file
in ./data/01_raw before running the script.

The script reports:
- the average precision of the model on the evaluation data:
  "Average precision: 0.xxxxxx"
- the size of the model:
  "Model size:       XXXX bytes"
- The total time spent in preprocessing / feature extraction:
  "Total * n_windows = XXXXX"
- The total time spent in model inference:
  "total number of microseconds, XXXX"  (Note: the second time this line appears, the first one relates to preprocessing)
- the RAM usage of the model on the embedded target:
  "[RecordingMicroAllocator] Arena allocation total XXXX bytes"
"""

import yaml
import json
import logging
import shutil
import tempfile
import zipfile
import ai_edge_litert

from pathlib import Path
from tempfile import TemporaryDirectory

import click
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.python.keras.utils.np_utils import to_categorical

from config import load_config
from data_preprocessing import run_data_preprocessing
#from embedded_code_generation import create_target, generate_and_flash
from feature_extraction import run_feature_extraction
from paths import CLIPS_DIR, EVAL_CLIPS_DIR, EVAL_ZIP_PATH, GEN_CODE_DIR, KERAS_MODEL_PATH

logger = logging.getLogger("biodcase_tiny")
logger.setLevel(logging.INFO)


def prepare_eval_clips(eval_clips_dir: Path, eval_zip_path: Path):
  """
  prepare eval clips
  """

  # skip if already prepared
  if eval_clips_dir.exists(): return

  # todo:
  # implement this for new dataset? - maybe not required
  raise NotImplementedError()

  # zip check
  if not eval_zip_path.exists():
    raise FileNotFoundError(f"Can't find evaluation data at {eval_zip_path}")

  # tmp
  with TemporaryDirectory() as tmp_folder:
    tmp_folder = Path(tmp_folder)

    with zipfile.ZipFile(eval_zip_path, 'r') as zip_ref:
      zip_ref.extractall(tmp_folder)

    labels_path = tmp_folder / "Evaluation Set (renamed)" / "binary_labels.csv"
    df = pd.read_csv(labels_path)

    inner_zip_path = tmp_folder / "Evaluation Set (renamed)" / "Evaluation Set.zip"

    with zipfile.ZipFile(inner_zip_path, 'r') as zip_ref:
      zip_ref.extractall(eval_clips_dir)

    # Recreate structure used in training dataset to be able to reuse the already written pipeline code
    negatives_folder = eval_clips_dir / "Negatives"
    yellowhammer_folder = eval_clips_dir / "Yellowhammer"

    negatives_folder.mkdir(exist_ok=True)
    yellowhammer_folder.mkdir(exist_ok=True)

    # Loop through labels and move files accordingly
    for _, row in df.iterrows():
      filename = row['Filename']
      label = row['Label']
      source_file = Path(eval_clips_dir) / filename

      # Move to Negatives folder
      if label == 0:
        destination = negatives_folder / filename
        shutil.move(str(source_file), str(destination))
      # Move to Yellowhammer folder
      elif label == 1:
        destination = yellowhammer_folder / filename
        shutil.move(str(source_file), str(destination))
      else:
        raise FileNotFoundError(f"File {filename} not found in evaluation_clips")
    print(f"Evaluation clips dataset preparation complete.")


def make_reference_ds(features_prq_path, features_shape_json_path, reference_dataset_path, channel_order=None):
  """
  reference dataset
  """

  # feature shape
  with features_shape_json_path.open("r") as f:
    features_shape = json.load(f)

  # read data parquet
  data = pd.read_parquet(features_prq_path)

  # features
  features = np.array(data["features"].to_list()).reshape((-1, *features_shape, 1))

  # channel ordering
  if channel_order is not None: features = np.transpose(features, channel_order)

  # one hot labels
  one_hot_labels = to_categorical(data["label"], num_classes=11)

  # dataset
  dataset = tf.data.Dataset.from_tensor_slices((features, one_hot_labels)).batch(32)
  dataset.save(str(reference_dataset_path))


def eval_model(features_shape_json_path: Path, features_prq_path: Path, tflite_model_path: Path, channel_order=None):
  """
  eval model
  """

  # data 
  data = pd.read_parquet(features_prq_path)
  data = data[data["split"].isin(["test", None])]  # data split None: eval clips

  # feature shape
  with features_shape_json_path.open("r") as f:
    features_shape = json.load(f)

  # feature data
  input_data = np.array(data["features"].to_list()).reshape(-1, *features_shape, 1)

  # channel ordering
  if channel_order is not None: input_data = np.transpose(input_data, channel_order)

  from ai_edge_litert.interpreter import Interpreter

  # tflite interpreter (new one)
  #interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
  interpreter = ai_edge_litert.interpreter.Interpreter(model_path=str(tflite_model_path))
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

  # label preparation
  y_preds = np.array(y_preds)
  y_true = to_categorical(data["label"].values, num_classes=11)

  # average precision
  average_precision = keras.src.metrics.AUC(curve='PR')
  average_precision.update_state(y_true, y_preds)
  print(f"Average precision: {average_precision.result().numpy()}")


# --
# options

@click.command()
@click.option('--model-path', '-m', type=click.Path(exists=True, path_type=Path), required=True, help='Path to the model file (.keras or .tflite)')
@click.option('--config-path', '-c', type=click.Path(exists=True, path_type=Path), required=True, help='Path to the pipeline configuration YAML file')
@click.option('--dataset', '-d', type=click.Choice(['test', 'eval']), default='eval', help="Dataset to use for evaluation. Test dataset there to replicate participant's reported performance")
@click.option('--channel-order', type=str, default='0,1,2,3', help='Channel order as comma-separated values (e.g., "0,1,2,3")')
@click.option('--evaluate-embedded/--no-evaluate-embedded', default=True, help='Whether to evaluate on embedded target')
@click.option('--quantize', is_flag=True, help='Whether to quantize the model')


def run_evaluation(model_path, config_path, dataset, channel_order, evaluate_embedded, quantize):
  """
  run evaluation
  """

  # config
  config = load_config(config_path)
  
  # channel order
  channel_order = tuple(int(x) for x in channel_order.split(','))

  # dataset
  if dataset == "eval":
    clips_dir = EVAL_CLIPS_DIR
    prepare_eval_clips(eval_clips_dir=clips_dir, eval_zip_path=EVAL_ZIP_PATH,)
  else:
    clips_dir = CLIPS_DIR

  # temp dir
  with tempfile.TemporaryDirectory(prefix="biodcase") as tmpdir:

    # paths
    root_dir = Path(tmpdir)
    preproc_prq_path = root_dir / "preproc.parquet"
    features_prq_path = root_dir / "features.parquet"
    features_shape_json_path = root_dir / "features_shape.json"
    features_sample_plot_path = root_dir / "features_sample_plot.png"
    reference_dataset_path = root_dir / "reference_dataset"

    # preprocessing
    run_data_preprocessing(config, clips_dir=clips_dir, preproc_prq_path=preproc_prq_path)

    # feature extraction
    run_feature_extraction(
      config,
      preproc_prq_path=preproc_prq_path,
      features_prq_path=features_prq_path,
      features_sample_plot_dir=features_sample_plot_path,
      features_shape_json_path=features_shape_json_path,
    )

    # model 
    if model_path.suffix == ".keras":
      make_reference_ds(
        features_prq_path=features_prq_path,
        features_shape_json_path=features_shape_json_path,
        reference_dataset_path=reference_dataset_path,
        channel_order=channel_order,
      )

    # evaluation
    with tempfile.NamedTemporaryFile() as model_file:

      # tflite 
      tflite_model_path = Path(model_file.name)
      tf.lite.experimental.Analyzer.analyze(model_path=str(tflite_model_path))

      # eval
      eval_model(
        features_shape_json_path=features_shape_json_path,
        features_prq_path=features_prq_path,
        tflite_model_path=tflite_model_path,
        channel_order=channel_order,
      )

      # embedded deployment
      if evaluate_embedded: run_deploy_generated_code(config, gen_code_dir=GEN_CODE_DIR)

      #target = create_target(model_path, reference_dataset_path, config, quantize=quantize)
      #model_file.write(target.get_model_buf())
      #if evaluate_embedded:
      #  generate_and_flash(config, target, GEN_CODE_DIR)


def run_load_and_test_keras_model_prediction(model_path=KERAS_MODEL_PATH):
  """
  load and test keras model
  """

  # label dict path
  label_dict_path = Path(model_path.parent) / 'label_dict.yaml'

  # label dict
  label_dict = yaml.safe_load(open(label_dict_path, 'r'))['label_dict'] if label_dict_path.is_file() else None

  # load model
  model = keras.models.load_model(model_path)

  # get input shape
  input_shape = model.get_config()['layers'][0]['config']['batch_shape'][1:]

  # add batch dimension
  input_shape = (1,) + input_shape

  # generate test sample
  x = np.random.randn(*input_shape)

  # predict class
  y = model.predict(x)

  # max prediction
  y_hat = np.argmax(y, axis=-1)

  # info print
  print("test input: ", x.shape)
  print("prediction: ", y)
  print("argmax: {}".format(y_hat))
  print("labels: {}".format([list(label_dict.keys())[yi] if label_dict is not None else '?' for yi in y_hat]))



if __name__ == '__main__':
  """
  model evaluations
  """

  # load and test model
  run_load_and_test_keras_model_prediction(model_path=KERAS_MODEL_PATH)

  # run evaluation
  #run_evaluation()