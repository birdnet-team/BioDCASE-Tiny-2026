import tempfile
from pathlib import Path

import keras
import tensorflow as tf

from biodcase_tiny.embedded.esp_target import ESPTarget
from biodcase_tiny.embedded.esp_toolchain import ESP_IDF_v5_2
from biodcase_tiny.feature_extraction.feature_extraction import make_constants

import logging

OK_MODEL_PATH = Path(__file__).parent / 'models' / 'ok_model.keras'
logging.basicConfig(level=logging.INFO)


class TestBuild:

    def test_build(self):
        toolchain = ESP_IDF_v5_2("/dev/ttyACM0")
        model = keras.models.load_model(OK_MODEL_PATH)
        feature_config = make_constants(
            win_samples=1024, sample_rate=16000, window_scaling_bits=16,
            mel_n_channels=20, mel_low_hz=100, mel_high_hz=8000
        )

        # synthetic data, we don't care
        data = tf.random.normal(
            shape=(100, 32, 24, 32, 1),
            dtype=tf.float32
        )
        labels = tf.random.uniform(shape=(100, 1), maxval=2, dtype=tf.int32)
        reference_dataset = tf.data.Dataset.from_tensor_slices((data, labels))

        target = ESPTarget(model, feature_config, reference_dataset)
        target.validate()
        with tempfile.TemporaryDirectory() as src_path:
            src_path = Path(src_path)
            target.process_target_templates(src_path)
            toolchain.compile(src_path=src_path)
            toolchain.flash(src_path=src_path)
