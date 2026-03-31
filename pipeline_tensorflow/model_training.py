# --
# model training

import sys
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import json

from keras import Model
from keras.src.metrics import AUC
from keras.src.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path

from pipeline_tensorflow.config import Config, load_config
from pipeline_tensorflow.paths import FEATURES_PRQ_PATH, KERAS_MODEL_PATH, FEATURES_SHAPE_JSON_PATH, REFERENCE_DATASET_PATH, CM_FIG_PATH
from pipeline_tensorflow.model import create_model, train_model

# required package paths
[sys.path.append(p) for p in [str(Path(__file__).parent.parent)] if p not in sys.path]

from plots import plot_confusion_matrix


def set_seeds(seed):
  """
  set seeds, for experiment replications
  """
  tf.config.experimental.enable_op_determinism()
  keras.utils.set_random_seed(seed)


def make_tf_datasets(data: pd.DataFrame, features_shape, label_dict, buffer_size=10000, seed=42, batch_size=32) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
  """
  tensorflow dataset
  """

  # split dict
  splits = {}

  # go trough feature data
  for split, group_data in data.groupby("split"):

    # features: shape (tf backend): batches, rows, cols, channels
    features = np.array(group_data["features"].to_list()).reshape((-1, *features_shape, 1))

    # one hot labels
    one_hot_labels = to_categorical(group_data["label"], num_classes=len(label_dict))

    # dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, one_hot_labels))
    dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed).batch(batch_size)
    splits[split] = dataset

  reference_dataset = splits["train"].shuffle(10000).take(100)
  return splits["train"], splits["validation"], reference_dataset


def get_class_weight(train_ds):
  """
  class weights
  """
  train_labels = train_ds["label"]
  l_counts: dict[str, int] = dict(train_labels.value_counts())
  tot_counts = len(train_labels)
  class_weight = {k: tot_counts / v for k, v in l_counts.items()}
  return class_weight


def predict_validation(model: Model, val_dataset: tf.data.Dataset):
  """
  predict validation
  """
  val_ds = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
  y_true = np.concat(list(val_ds.map(lambda x, y: y).as_numpy_iterator()))
  y_pred = model.predict(val_ds)
  return y_true, y_pred


def run_model_training(config: Config):
  """
  model training
  """

  # assertions
  assert FEATURES_PRQ_PATH.is_dir(), "No feature extractions available in [{}], run feature extraction first!".format(FEATURES_PRQ_PATH)

  # create path if not already created
  if not KERAS_MODEL_PATH.parent.is_dir(): KERAS_MODEL_PATH.parent.mkdir()
  if not CM_FIG_PATH.parent.is_dir(): CM_FIG_PATH.parent.mkdir()

  # seed for reproducibility
  set_seeds(config.model_training.seed)

  # feature shape
  features_shape = json.load(open(FEATURES_SHAPE_JSON_PATH, 'r'))

  # data
  data = pd.read_parquet(FEATURES_PRQ_PATH)

  # load class dict
  label_dict = yaml.safe_load(open(FEATURES_PRQ_PATH.parent / 'label_dict.yaml'))['label_dict']

  # got flattened when writing parquet, restore shape now
  data["features"] = data["features"].apply(lambda x: x.reshape(features_shape))

  # class weights
  class_weight = get_class_weight(data[data["split"] == "train"])
  train_ds, valid_ds, reference_ds = make_tf_datasets(
    data,
    features_shape,
    label_dict,
    config.model_training.shuffle_buff_n, 
    config.model_training.seed,
    config.model_training.batch_size
  )

  # model creation
  model = create_model((*features_shape, 1), num_output_classes=len(label_dict))

  # model training
  model = train_model(model, train_ds, valid_ds, config, class_weight)

  # model validation
  y_true, y_pred = predict_validation(model, valid_ds)

  # save model, reference dataset, and label dict
  model.save(KERAS_MODEL_PATH)
  reference_ds.save(str(REFERENCE_DATASET_PATH))
  yaml.dump({'label_dict': label_dict}, open(KERAS_MODEL_PATH.parent / 'label_dict.yaml', 'w'))

  # info
  print("Training finished successfully!\nConfusion matrix saved in [{}].".format(CM_FIG_PATH))

  # plot confusion matrix
  plot_confusion_matrix(y_true, y_pred, labels=list(label_dict.keys()), plot_path=CM_FIG_PATH, show_plot_flag=False, apply_argmax=True)


if __name__ == "__main__":
  """
  model training
  """

  # config
  config = load_config()

  # run model training
  run_model_training(config)