# --
# model - tensorflow

from keras import Model, layers
from keras.src.applications.mobilenet import _conv_block, _depthwise_conv_block
from keras.src.callbacks import History, EarlyStopping, TensorBoard
from keras.src.metrics import AUC
import tensorflow as tf

from paths import TENSORBOARD_LOGS_PATH
from config import Config


def create_model(input_shape, num_output_classes, n_filters_1=32, n_filters_2=64, dropout=0.02) -> Model:
  """
  create model
  """

  # input
  inputs = layers.Input(shape=input_shape)

  # layers
  x = _conv_block(inputs, filters=n_filters_1, alpha=1, kernel=(10, 4), strides=(5, 2))
  x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_1, alpha=1, block_id=1)
  x = layers.GlobalMaxPooling2D(keepdims=True)(x)
  x = layers.Dropout(dropout, name="dropout1")(x)
  x = layers.Flatten()(x)

  # output
  x = layers.Dense(num_output_classes)(x)
  outputs = layers.Softmax()(x)

  # model
  model = Model(inputs, outputs, name="mobilenet_slimmed")
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC(curve='PR', name='ap')])
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