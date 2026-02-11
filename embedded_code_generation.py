#   Copyright 2025 BirdNET-Team
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from pathlib import Path

import keras
import tensorflow as tf
from keras import Model

from biodcase_tiny.embedded.esp_target import ESPTarget
from biodcase_tiny.embedded.esp_toolchain import ESP_IDF_v5_2
from biodcase_tiny.feature_extraction.feature_extraction import make_constants
from config import Config, load_config
from paths import KERAS_MODEL_PATH, REFERENCE_DATASET_PATH, GEN_CODE_DIR, TFLITE_MODEL_PATH


def run_embedded_code_generation(config: Config, model_path: Path = KERAS_MODEL_PATH, reference_dataset_path: Path = REFERENCE_DATASET_PATH, tflite_model_path: Path = TFLITE_MODEL_PATH, gen_code_dir: Path = GEN_CODE_DIR, quantize: bool = False):
    """
    embedded code generation
    """

    # create directory
    if not gen_code_dir.is_dir(): gen_code_dir.mkdir()

    # check model file ending
    if model_path.suffix == ".keras":
        model = keras.models.load_model(model_path)
        reference_dataset = tf.data.Dataset.load(str(reference_dataset_path))
    elif model_path.suffix == ".tflite":
        with model_path.open("rb") as f:
            model = f.read()
        reference_dataset = None
    else:
        raise ValueError("Only Keras and tflite format supported")

    # configs
    dp_c = config.data_preprocessing
    fe_c = config.feature_extraction
    feature_config = make_constants(sample_rate=dp_c.sample_rate, win_samples=fe_c.window_len, window_scaling_bits=fe_c.window_scaling_bits, mel_n_channels=fe_c.mel_n_channels, mel_low_hz=fe_c.mel_low_hz, mel_high_hz=fe_c.mel_high_hz, mel_post_scaling_bits=fe_c.mel_post_scaling_bits)

    # target createion
    target = ESPTarget(model, feature_config, reference_dataset, quantize=quantize)
    target.validate()

    # tflite model to buffer
    with tflite_model_path.open("wb") as f: f.write(target.get_model_buf())

    # source path
    src_path = gen_code_dir / "src"
    src_path.mkdir(exist_ok=True)

    # write templates
    target.process_target_templates(src_path)

    # toolchain: compile, flash, and monitor
    toolchain = ESP_IDF_v5_2(config.embedded_code_generation.serial_device)
    toolchain.compile(src_path=src_path)
    toolchain.flash(src_path=src_path)
    toolchain.monitor(src_path=src_path)


if __name__ == '__main__':
    """
    embedded code generation
    """

    # config
    config = load_config()

    # run embedded code generation
    run_embedded_code_generation(config)
