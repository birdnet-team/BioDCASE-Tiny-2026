import re
import yaml
import numpy as np
from pathlib import Path

from ai_edge_quantizer import quantizer, recipe
from ai_edge_quantizer.utils import tfl_interpreter_utils

# TODO fix data methods
def filter_files_with_config(files, cfg_filter_files={'is_used': False, 're_contains': '.*'}):
    """
    filter files with config {filter_files: {is_used, re_contains}}
    """

    # returns
    if cfg_filter_files is None: return files
    if not cfg_filter_files['is_used']: return files

    # assert string in re
    assert isinstance(cfg_filter_files['re_contains'], str)

    # filter files
    filtered_files = [f for f in files if re.search(cfg_filter_files['re_contains'], str(f))]

    # assert that there are still some files left
    assert len(filtered_files), 'No files left after filtering!!! Change in config.yaml -> filter_files.re_contains'

    # filter files
    return filtered_files

def loadCalibrationDataset(cfg, key="train", cache_id=None):
    """
    load data from cache for calibration
    """

    additional_file_filter_cfg={'is_used': True, 're_contains': key}

    # target cached path
    target_cached_path = Path(cfg['caching']['root_path']) / cfg['caching']['cache_id']

    # target cache id 
    if not cache_id is None:

      # change cached path
      cached_path=Path(cfg['caching']['root_path']) / cfg['caching']['cache_id']
      target_cached_path = cached_path.parent / cache_id

      # check if path exists
      if not target_cached_path.exists(): raise ValueError('cach_id: {} does not exist in path: {}!'.format(cache_id, cached_path.parent))

    # cached files
    cached_files_filtered = filter_files_with_config(sorted(list(target_cached_path.glob('**/*.npz'))), cfg['load_cache']['filter_files'])

    # additional filtering
    cached_files_filtered = filter_files_with_config(cached_files_filtered, additional_file_filter_cfg)


    cache_info=yaml.safe_load(open(str(target_cached_path / 'cache_info.yaml')))

    calibration_samples = []

    for i, cached_file in enumerate(cached_files_filtered):
      # data 
      data = np.load(cached_file)
      x = data['x'].reshape(cache_info['feature_size_origin'])

      if not cfg['feature_handler_add_kwargs']['add_channel_dimension']: x = x[np.newaxis, :]

      calibration_samples.append({
            'args_0': x
        })
    #print(len(calibration_samples))  
    calibration_data = {
        tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: calibration_samples,
    }
          
    # success
    print("Calibration dataset loaded successfully!")
    return calibration_data


def model_quantization(cfg, tfliteFP32path, tfliteINT8path):
    calibration_data=loadCalibrationDataset(cfg)
    qt_static = quantizer.Quantizer(tfliteFP32path)
    qt_static.load_quantization_recipe(recipe.static_wi8_ai8())
    calibration_result = qt_static.calibrate(calibration_data)
    qt_static.quantize(calibration_result).export_model(tfliteINT8path, overwrite=True)


if __name__ == '__main__':
    # yaml config file
  model_quantization("output/03_model/ModelTinyMl.tflite", "output/03_model/ModelTinyMlINT8.tflite")