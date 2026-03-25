#   Copyright 2024 BirdNET-Team
#   Copyright 2024 fold ecosystemics
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

"""
Build target for korvo2_bird_logger template project.
The class will check that the operations from the input keras model are supported by tflite-micro.
It then converts the keras model to tflite, and prepares all the information needed by the project template
to generate the final code.
"""

import os
import shutil
import jinja2
import importlib

from copy import deepcopy
from pathlib import Path
from biodcase_tiny.feature_extraction.feature_extraction import FeatureConstants, convert_constants


class ESPTarget():
  """
  esp target - src code preparation for esp32-s3 with templates - jinja
  """

  def __init__(self, template_dir: Path, model_path: Path, feature_config: FeatureConstants, reference_dataset_path: Path | None = None, quantize: bool = False):

    # arguments
    self.template_dir = template_dir
    self.model_path = model_path
    self.feature_config = feature_config
    self.reference_dataset_path = reference_dataset_path
    self.quantize = quantize

    # assertions
    assert self.template_dir.is_dir(), "Check template directory agian: {}".format(self.template_dir)

    # members
    self._model_buf = self.create_model_buffer()
    self._model_ops = self.get_model_ops_and_acts(self._model_buf)
    self._feature_config_buf = convert_constants(feature_config)

    # other members
    self.env = None


  def create_model_buffer(self):
    """
    create model buffer
    """

    # is it a keras model?
    if self.model_path.suffix == ".keras": return self._create_model_buf_from_keras_model()

    # is it a tflite model?
    if self.model_path.suffix == ".tflite":

      # simply read the data as bytes
      with self.model_path.open("rb") as f: model = f.read()

      # quantize flag check
      if self.quantize: raise ValueError("A tflite model was provided but `quantize` set to True.")

      # model buffer
      return deepcopy(model)

    # all other suffixes are not supported
    raise ValueError("Either a keras or tflite model is required, yours: {}".format(self.model_path))


  def validate(self):
    """
    Validate Target inputs, including the compatibility of model.
    """

    # no tensorflow installed no model check
    if self._model_ops is None: 
      print("***No model check done!")
      return

    # check model compatibility
    if None in self._model_ops:
      raise ValueError(
        "Model contains op(s) that can't be converted to tflite micro. "
        f"Known ops: {self._model_ops.difference({None})}"
      )


  def process_target_templates(self, outdir: Path, template_extension="jinja") -> None:
    """
    process target templates
    """

    # validate first
    self.validate()

    # get available projects and their root template folders
    self.env = jinja2.Environment(loader=jinja2.FileSystemLoader(self.template_dir))

    # extract context to be passed to jinja render
    context = self.extract_context_for_jinja_renderer()

    # Render and save each template
    if not outdir.exists():
      raise ValueError(f"{str(outdir)} does not exist, please create it.")

    # Process each file in the template directory
    for template_name in self.env.list_templates():
      template_path = Path(template_name)
      if template_path.suffix.lstrip(".") == template_extension:
        # Render and save the template file
        template = self.env.get_template(template_name)
        output_path = outdir / template_path.with_suffix("")
        with output_path.open("w") as f:
          f.write(template.render(context))
      else:
        # Copy non-template files directly
        src_path = self.template_dir / template_name
        dst_path = outdir / template_name
        os.makedirs(dst_path.parent, exist_ok=True)
        shutil.copyfile(src_path, dst_path)


  def _create_model_buf_from_keras_model(self):
    """
    create model buffer
    """

    # keras and tensorflow package required
    import keras
    import tensorflow as tf

    # load keras model
    model = keras.models.load_model(self.model_path)

    # reference dataset available?
    if self.reference_dataset_path is None: raise ValueError("Reference dataset must be provided when a keras model is passed.")

    # load reference dataset
    reference_dataset = tf.data.Dataset.load(str(self.reference_dataset_path))

    # converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] if self.quantize else []
    if self.quantize:
      converter.inference_input_type = tf.dtypes.int8
      converter.inference_output_type = tf.dtypes.int8
      converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
      converter._experimental_disable_per_channel_quantization_for_dense_layers = True

    def representative_dataset_gen():
      for example_spectrograms, example_spect_labels in reference_dataset.take(10):
        for X, _ in zip(example_spectrograms, example_spect_labels):
          # Add a `batch` dimension, so that the spectrogram can be used
          yield [X[tf.newaxis, ...]]

    converter.representative_dataset = representative_dataset_gen
    model_buf = converter.convert()
    return model_buf


  def extract_context_for_jinja_renderer(self) -> dict:
    """
    extract context - for jinja renderer
    """
    return {
      "feature_config": {"hex_vals": [hex(b) for b in self._feature_config_buf]},
      "model": {"hex_vals": [hex(b) for b in self._model_buf]},
      }


  def save_tflite_model(self, outdir: Path) -> None:
    """
    save tflite
    """
    with outdir.open("wb") as f:
      f.write(self._model_buf)


  def get_model_ops_and_acts(self, model_buf):
    """
    Extracts a set of operators from a tflite model.
    This fn is adapted from tensorflow lite micro tools scripts:
    (https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver/generate_micro_mutable_op_resolver_from_model.py)
    """

    # skip if tensorflow is not installed
    if not bool(importlib.util.find_spec('tensorflow')): 
      print("\n***@esp_target.get_model_ops_and_acts: tensorflow not installed! -> skip model op check")
      return None

    from tensorflow.lite.tools import visualize as tflite_vis

    custom_op_found = False
    operators_and_activations = set()
    data = tflite_vis.CreateDictFromFlatbuffer(model_buf)

    # operation check
    for op_code in data["operator_codes"]:
      if op_code["custom_code"] is None:
        op_code["builtin_code"] = max(op_code["builtin_code"], op_code["deprecated_builtin_code"])
      else:
        custom_op_found = True
        operators_and_activations.add(tflite_vis.NameListToString(op_code["custom_code"]))

    # double check for custom operations
    for op_code in data["operator_codes"]:

      # Custom operator already added.
      if custom_op_found and tflite_vis.BuiltinCodeToName(op_code["builtin_code"]) == "CUSTOM":
        continue

      # will be None if unknown
      operators_and_activations.add(tflite_vis.BuiltinCodeToName(op_code["builtin_code"]))
    return operators_and_activations


  def get_model_buf(self): return self._model_buf


# --
# not used code
#
# def tflite_to_byte_array(tflite_file: Path):
#     with tflite_file.open("rb") as input_file:
#         buffer = input_file.read()
#     return buffer


# def parse_op_str(op_str):
#     """Converts a flatbuffer operator string to a format suitable for Micro
#     Mutable Op Resolver. Example: CONV_2D --> AddConv2D.

#     This fn is adapted from tensorflow lite micro tools scripts:
#     (https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver/generate_micro_mutable_op_resolver_from_model.py)
#     """
#     import re
#     # Edge case for AddDetectionPostprocess().
#     # The custom code is TFLite_Detection_PostProcess.
#     op_str = op_str.replace("TFLite", "")
#     word_split = re.split("[_-]", op_str)
#     formatted_op_str = ""
#     for part in word_split:
#         if len(part) > 1:
#             if part[0].isalpha():
#                 formatted_op_str += part[0].upper() + part[1:].lower()
#             else:
#                 formatted_op_str += part.upper()
#         else:
#             formatted_op_str += part.upper()
#     # Edge cases
#     formatted_op_str = formatted_op_str.replace("Lstm", "LSTM")
#     formatted_op_str = formatted_op_str.replace("BatchMatmul", "BatchMatMul")
#     return formatted_op_str