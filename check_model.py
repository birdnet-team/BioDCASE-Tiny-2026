import tensorflow as tf

from paths import TFLITE_MODEL_PATH


if __name__ == "__main__":
    tf.lite.experimental.Analyzer.analyze(model_path=str(TFLITE_MODEL_PATH))
