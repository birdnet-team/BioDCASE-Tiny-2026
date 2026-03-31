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
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path

from pipeline_tensorflow.config import Config, load_config
from pipeline_tensorflow.paths import KERAS_MODEL_PATH, REFERENCE_DATASET_PATH, CM_FIG_PATH, TFLITE_MODEL_PATH
from pipeline_tensorflow.model import create_model, train_model

from keras.src.utils import to_categorical

# required package paths
[sys.path.append(p) for p in [str(Path(__file__).parent.parent)] if p not in sys.path]

from plots import plot_confusion_matrix


def set_seeds(seed):
  """
  set seeds, for experiment replications
  """
  tf.config.experimental.enable_op_determinism()
  keras.utils.set_random_seed(seed)

def get_class_weight(train_labels):
  """
  class weights
  """
  unique, counts = np.unique(train_labels.astype(int), return_counts=True)
  l_counts: dict[str, int] = dict(zip(unique, counts))
  tot_counts = len(train_labels)
  class_weight = {k: tot_counts / v for k, v in l_counts.items()}
  return class_weight

def tf_dataset(datamodule, features_shape, shuffle, buffer_size=None, seed=None, batch_size=None):
    
    # features: shape (tf backend): batches, rows, cols, channels
    features = datamodule.features.reshape(-1, *features_shape)

    # one hot labels
    one_hot_labels = to_categorical(datamodule.targets)

    # dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, one_hot_labels))
    if shuffle:
      dataset = dataset.shuffle(
          buffer_size=buffer_size,
          seed=seed,
          reshuffle_each_iteration=True
      )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def predict_validation(model: Model, val_dataset: tf.data.Dataset):
  """
  predict validation
  """
  val_ds = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
  y_true = np.concat(list(val_ds.map(lambda x, y: y).as_numpy_iterator()))
  y_pred = model.predict(val_ds)
  return y_true, y_pred

def convert_model(keras_path, tflite_path):
  # Code taken from students
  model = tf.keras.models.load_model(keras_path)

  # Convert to TFLite with quantization
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  # Save the converted model to a .tflite file
  with open(tflite_path, 'wb') as f:
      f.write(tflite_model)


def run_model_training(config: Config, datamodule_train, datamodule_validation):
  """
  model training
  """

  # assertions
  
  # create path if not already created
  if not KERAS_MODEL_PATH.parent.is_dir(): KERAS_MODEL_PATH.parent.mkdir()
  if not CM_FIG_PATH.parent.is_dir(): CM_FIG_PATH.parent.mkdir()

  # seed for reproducibility
  set_seeds(config.model_training.seed)

  # feature shape
  c, w, h = datamodule_train.get_feature_shape_at_load()
  features_shape = (w, h, c)
  print(features_shape)

  # data
  train_ds = tf_dataset(datamodule_train, features_shape,
    shuffle=True,
    buffer_size=config.model_training.shuffle_buff_n, 
    seed=config.model_training.seed,
    batch_size=config.model_training.batch_size
  )
  valid_ds = tf_dataset(datamodule_validation, features_shape,
    shuffle=True,
    buffer_size=config.model_training.shuffle_buff_n, 
    seed=config.model_training.seed,
    batch_size=config.model_training.batch_size
  )
  reference_ds = train_ds.shuffle(10000).take(100)

  # load class dict
  label_dict = datamodule_train.get_label_dict()

  # class weights
  class_weight = get_class_weight(datamodule_train.targets)


  # model creation
  model = create_model(features_shape, num_output_classes=len(label_dict))
  model.summary()

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

  convert_model(KERAS_MODEL_PATH, TFLITE_MODEL_PATH)

if __name__ == "__main__":
  """
  model training
  """

  # config
  config = load_config()

  # run model training
  run_model_training(config)