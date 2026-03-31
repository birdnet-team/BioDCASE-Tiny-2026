# --
# model - tensorflow

from keras import Model, layers, optimizers, losses, metrics
from keras.src.callbacks import EarlyStopping, TensorBoard
import tensorflow as tf

from pipeline_tensorflow.paths import TENSORBOARD_LOGS_PATH
from pipeline_tensorflow.config import Config


def create_model(input_shape, num_output_classes, n_filters_1=32, dropout=0.05) -> Model:
  """
  create model
  """

  # input
  inputs = layers.Input(shape=input_shape)

  x = layers.Conv2D(filters=n_filters_1, kernel_size=3, activation='relu')(inputs)
  x = layers.MaxPooling2D(pool_size=2)(x)

  x = layers.Conv2D(filters=n_filters_1*2, kernel_size=3, activation='relu')(x)
  x = layers.MaxPooling2D(pool_size=4)(x)

  x = layers.Conv2D(filters=n_filters_1*4, kernel_size=3, activation='relu')(x)
  x = layers.GlobalAveragePooling2D()(x)

  x = layers.Dropout(dropout, name="dropout1")(x)
  x = layers.Dense(32, activation='relu')(x)

  # output
  x = layers.Dense(num_output_classes)(x)
  outputs = layers.Softmax()(x)

  # model
  model = Model(inputs, outputs, name="cnn_baseline")
  model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=losses.CategoricalCrossentropy(),
    metrics=[metrics.CategoricalAccuracy()]
    )

  return model


def train_model(model: Model, train_ds, valid_ds, config: Config, class_weight) -> Model:
  """
  train model
  """
  print("train start")
  tr_cfg = config.model_training
  train_ds = train_ds.cache().shuffle(tr_cfg.shuffle_buff_n).prefetch(tf.data.AUTOTUNE)
  valid_ds = valid_ds.cache().prefetch(tf.data.AUTOTUNE)
  model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=tr_cfg.n_epochs,
    class_weight=class_weight,
    callbacks=[
      EarlyStopping(patience=tr_cfg.early_stopping.patience),
      TensorBoard(TENSORBOARD_LOGS_PATH, update_freq=1)
      ]
  )
  print("train end")
  return model