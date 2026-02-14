# --
# model training

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.src.metrics import AUC
from keras.src.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay

import yaml
import json
from config import Config, load_config
from paths import FEATURES_PRQ_PATH, KERAS_MODEL_PATH, FEATURES_SHAPE_JSON_PATH, REFERENCE_DATASET_PATH, CM_FIG_PATH
from model import create_model, train_model


def set_seeds(seed):
  """
  set seeds, for experiment replecations
  """
  tf.config.experimental.enable_op_determinism()
  keras.utils.set_random_seed(seed)


def make_tf_datasets(data: pd.DataFrame, features_shape, class_dict, buffer_size=10000, seed=42, batch_size=32) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
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
    one_hot_labels = to_categorical(group_data["label"], num_classes=len(class_dict))

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


def plot_confusion_matrix(y_target, y_predicted, labels=None, plot_path=None, is_showing_plot=False):
  """
  plot confusion matrix
  """

  # packages
  import sklearn
  import matplotlib.pyplot as plt
  import matplotlib
  from mpl_toolkits.axes_grid1 import make_axes_locatable

  # argmax targets and predictions
  y_target = np.argmax(y_target, axis=1)
  y_predicted = np.argmax(y_predicted, axis=1)

  # labels
  labels = labels if not labels is None and not len(np.unique([y_target, y_predicted])) != len(labels) else np.unique([y_target, y_predicted])

  # confusion matrix
  confusion_matrix = sklearn.metrics.confusion_matrix(y_target, y_predicted)

  # max value
  max_value = int(np.max(np.sum(confusion_matrix, axis=1)))

  # create figure
  fig = plt.figure(figsize=(12, 8))

  # create axis
  ax = fig.add_subplot(1, 1, 1)

  # kwargs
  imshow_kwargs = {'origin': 'upper', 'aspect': 'equal', 'interpolation': 'none', 'cmap': 'OrRd', 'vmax': max_value, 'vmin': 0}

  # image
  im = ax.imshow(confusion_matrix, **imshow_kwargs)

  # text handling
  for y_pred_pos in range(len(labels)):
    for y_true_pos in range(len(labels)):

      # font color and size
      font_color = 'black' if confusion_matrix[y_pred_pos, y_true_pos] < 0.6 * max_value else 'white'
      fontsize = 8 if len(labels) > 10 else 11

      # write numbers inside
      ax.text(y_true_pos, y_pred_pos, confusion_matrix[y_pred_pos, y_true_pos], ha='center', va='center', color=font_color, fontsize=fontsize, bbox=None)

  # care about labels
  ax.set_xticks(np.arange(len(labels)))
  ax.set_yticks(np.arange(len(labels)))
  ax.set_xticklabels(labels)
  ax.set_yticklabels(labels)
  ax.set_xlabel('Predicted Labels', fontsize=13)
  ax.set_ylabel('True Labels', fontsize=13)
  ax.set_title('Confusion Matrix')
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

  # spacings
  fig.subplots_adjust(bottom=0.25)

  # devider for cax
  cax = make_axes_locatable(plt.gca()).append_axes('right', size='3%', pad='3%')

  # colorbar
  color_bar = fig.colorbar(im, cax=cax)
  color_bar.ax.tick_params(labelsize=10)

  # save figure
  if plot_path is not None: fig.savefig(plot_path, dpi=100)

  # show plot or close
  plt.show() if is_showing_plot else plt.close()


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
  class_dict = yaml.safe_load(open(FEATURES_PRQ_PATH.parent / 'class_dict.yaml'))['class_dict']

  # got flattened when writing parquet, restore shape now
  data["features"] = data["features"].apply(lambda x: x.reshape(features_shape))

  class_weight = get_class_weight(data[data["split"] == "train"])
  train_ds, valid_ds, reference_ds = make_tf_datasets(
    data,
    features_shape,
    class_dict,
    config.model_training.shuffle_buff_n, 
    config.model_training.seed,
    config.model_training.batch_size
  )

  # model creatino
  model = create_model((*features_shape, 1), num_output_classes=len(class_dict))

  # model training
  model = train_model(model, train_ds, valid_ds, config, class_weight)

  # model validation
  y_true, y_pred = predict_validation(model, valid_ds)

  # save model
  model.save(KERAS_MODEL_PATH)
  reference_ds.save(str(REFERENCE_DATASET_PATH))

  # info
  print("Training finished successfully!\nConfusion matrix saved in [{}].".format(CM_FIG_PATH))

  # plot confusion matrix
  plot_confusion_matrix(y_true, y_pred, labels=list(class_dict.keys()), plot_path=CM_FIG_PATH, is_showing_plot=False)


if __name__ == "__main__":
  """
  model training
  """

  # config
  config = load_config()

  # run model training
  run_model_training(config)