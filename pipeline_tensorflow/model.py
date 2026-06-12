# --
# model - tensorflow

from keras import Model, layers, optimizers, losses, metrics
from keras.src.callbacks import EarlyStopping, TensorBoard
import tensorflow as tf

from pipeline_tensorflow.paths import TENSORBOARD_LOGS_PATH


def create_model(input_shape, num_output_classes, n_filters_1=16, dropout=0.05) -> Model:
  """
  create model
  """

  input_layer = tf.keras.layers.Input(shape=input_shape)
  conv1d_1 =    tf.keras.layers.Conv2D(filters=8, kernel_size=7, strides=(1,4), activation='relu')(input_layer)
  output_main = tf.keras.layers.MaxPooling2D(pool_size=2) (conv1d_1)

  output_main = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu') (output_main)
  output_main = tf.keras.layers.MaxPooling2D(pool_size=2) (output_main)

  output_main = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu') (output_main)
  #output_main = tf.keras.layers.MaxPooling2D(pool_size=2) (output_main)

  output_main = tf.keras.layers.GlobalAveragePooling2D()(output_main)
  #output_main = tf.keras.Dropout(0.02, name="dropout1")(output_main)
  output_main = tf.keras.layers.Dense(units=128, activation='relu') (output_main)
  output_main = tf.keras.layers.Dense(units=num_output_classes, activation='softmax', name='output_main') (output_main)

  model = tf.keras.models.Model(inputs=input_layer, outputs=output_main)
  
  model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=losses.CategoricalCrossentropy(),
    metrics=[metrics.CategoricalAccuracy()]
    )

  return model


def train_model(model: Model, train_ds, valid_ds, tr_cfg, class_weight) -> Model:
  """
  train model
  """
  print("train start")
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