import re
import yaml
from datamodule import DatamoduleTinyMl

from ai_edge_quantizer import quantizer, recipe

from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer.utils import tfl_interpreter_utils

def clean_input_layer_name(name: str) -> str:
    name = name.split(":")[0]
    name = re.sub(r"^serving_default_", "", name)
    return name

def model_quantization(cfg_datamodule, tfliteFP32path, tfliteINT8path):
    interpreter = Interpreter(model_path=str(tfliteFP32path))

    input_details = interpreter.get_input_details()

    datamodule_calibrate = DatamoduleTinyMl(cfg_datamodule, load_set_on_init='test')

    calibration_samples=[]

    for i, sample in enumerate(datamodule_calibrate.features):
      calibration_samples.append({
            clean_input_layer_name(input_details[0]['name']): sample
        })

    calibration_data = {
        tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: calibration_samples,
    }

    qt_static = quantizer.Quantizer(tfliteFP32path)
    qt_static.load_quantization_recipe(recipe.static_wi8_ai8())
    calibration_result = qt_static.calibrate(calibration_data)
    qt_static.quantize(calibration_result).export_model(tfliteINT8path, overwrite=True)


if __name__ == '__main__':
  # yaml config file
  cfg_datamodule = yaml.safe_load(open('./config.yaml'))['datamodule']
  model_quantization(cfg_datamodule, "output/03_models/Baseline.tflite", "output/03_models/BaselineINT8.tflite")