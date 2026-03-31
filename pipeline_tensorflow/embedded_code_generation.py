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

import sys
import keras
import tensorflow as tf
from keras import Model

from pathlib import Path
from pipeline_tensorflow.config import Config, load_config
from pipeline_tensorflow.paths import KERAS_MODEL_PATH, REFERENCE_DATASET_PATH, GEN_CODE_DIR, TFLITE_MODEL_PATH, TEMPLATE_DIR

# required package paths
[sys.path.append(p) for p in [str(Path(__file__).parent.parent)] if p not in sys.path]

from biodcase_tiny.embedded.esp_target import ESPTarget
from biodcase_tiny.embedded.esp_toolchain import ESPToolchain
from biodcase_tiny.feature_extraction.feature_extraction import make_constants


def run_embedded_code_generation(config: Config, model_path: Path, reference_dataset_path: Path, tflite_model_path: Path, gen_code_dir: Path, quantize: bool = False):
  """
  embedded code generation
  """

  # create directory
  if not gen_code_dir.is_dir(): gen_code_dir.mkdir()

  # configs
  dp_c = config.data_preprocessing
  fe_c = config.feature_extraction
  feature_config = make_constants(sample_rate=dp_c.sample_rate, win_samples=fe_c.window_len, window_scaling_bits=fe_c.window_scaling_bits, mel_n_channels=fe_c.mel_n_channels, mel_low_hz=fe_c.mel_low_hz, mel_high_hz=fe_c.mel_high_hz, mel_post_scaling_bits=fe_c.mel_post_scaling_bits)

  # info
  print("\nTarget creation:")

  # target creation, validation, and saving of tflite model
  target = ESPTarget(TEMPLATE_DIR, model_path, feature_config, reference_dataset_path, quantize=quantize)
  target.validate()
  target.save_tflite_model(tflite_model_path)

  # source path
  src_path = gen_code_dir / "src"
  src_path.mkdir(exist_ok=True)

  # write templates
  target.process_target_templates(src_path)

  # info
  print("\nCode Compilation:")

  # toolchain: compile, flash, and monitor
  toolchain = ESPToolchain(config.embedded_code_generation.serial_device)
  #toolchain.set_target(src_path=src_path)
  toolchain.compile(src_path=src_path)

  # info
  print("Embedded code has successfully been generated!")


if __name__ == '__main__':
  """
  embedded code generation
  """

  # config
  config = load_config()

  # run embedded code generation
  #run_embedded_code_generation(config)
  run_embedded_code_generation(config, model_path=KERAS_MODEL_PATH, reference_dataset_path=REFERENCE_DATASET_PATH, tflite_model_path=TFLITE_MODEL_PATH, gen_code_dir=GEN_CODE_DIR)