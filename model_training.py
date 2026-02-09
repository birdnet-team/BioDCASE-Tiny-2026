import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.src.metrics import AUC
from keras.src.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay

import json
from config import Config, load_config
from paths import FEATURES_PRQ_PATH, KERAS_MODEL_PATH, FEATURES_SHAPE_JSON_PATH, REFERENCE_DATASET_PATH, CM_FIG_PATH
from model import create_model, train_model


def set_seeds(seed):
    tf.config.experimental.enable_op_determinism()
    keras.utils.set_random_seed(seed)


def make_tf_datasets(
    data: pd.DataFrame,
    features_shape,
    buffer_size=10000,
    seed=42,
    batch_size=32,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    splits = {}
    for split, group_data in data.groupby("split"):
        # shape (tf backend): batches, rows, cols, channels
        features = np.array(group_data["features"].to_list()).reshape((-1, *features_shape, 1))
        one_hot_labels = to_categorical(group_data["label"], num_classes=2)
        dataset = tf.data.Dataset.from_tensor_slices((features, one_hot_labels))
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed).batch(batch_size)
        splits[split] = dataset

    reference_dataset = splits["train"].shuffle(10000).take(100)
    return splits["train"], splits["validation"], reference_dataset


def get_class_weight(train_ds):
    train_labels = train_ds["label"]
    l_counts: dict[str, int] = dict(train_labels.value_counts())
    tot_counts = len(train_labels)
    class_weight = {k: tot_counts / v for k, v in l_counts.items()}
    return class_weight


def predict_validation(model: Model, val_dataset: tf.data.Dataset):
    val_ds = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
    y_true = np.concat(list(val_ds.map(lambda x, y: y).as_numpy_iterator()))
    y_pred = model.predict(val_ds)
    return y_true, y_pred


def get_confusion_matrix(y_true, y_pred, labels: list[str]):
    return ConfusionMatrixDisplay.from_predictions(
        np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1),
        display_labels=labels
    ).figure_


def run_model_training(config: Config):
    set_seeds(config.model_training.seed)  # for reproducibility
    with open(FEATURES_SHAPE_JSON_PATH, "r") as f:
        features_shape = json.load(f)

    data = pd.read_parquet(FEATURES_PRQ_PATH)
    data["features"] = data["features"].apply(lambda x: x.reshape(features_shape))  # got flattened when writing parquet, restore shape now

    class_weight = get_class_weight(data[data["split"] == "train"])
    train_ds, valid_ds, reference_ds = make_tf_datasets(
        data,
        features_shape,
        config.model_training.shuffle_buff_n, 
        config.model_training.seed,
        config.model_training.batch_size
    )
    model = create_model((*features_shape, 1))
    model = train_model(model, train_ds, valid_ds, config, class_weight)
    y_true, y_pred = predict_validation(model, valid_ds)
    cm_fig = get_confusion_matrix(y_true, y_pred, labels=["Other", "Yellowhammer"])
    cm_fig.savefig(CM_FIG_PATH)
    model.save(KERAS_MODEL_PATH)
    reference_ds.save(str(REFERENCE_DATASET_PATH))


if __name__ == "__main__":
    config = load_config()
    run_model_training(config)