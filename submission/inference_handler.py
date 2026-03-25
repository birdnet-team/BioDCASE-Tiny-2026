# --
# inference handler

import yaml
import torch
import numpy as np
import sys
import importlib

from pathlib import Path


class InferenceHandler():
  """
  inference handler - handle your inferences for submission
  """

  def __init__(self, cfg={}, **kwargs):

    # arguments
    self.cfg = cfg
    self.kwargs = kwargs

    # init config
    self.cfg_init()

    # required package paths
    [sys.path.append(p) for p in self.cfg['add_python_paths'] if p not in sys.path]

    # members
    self.model = None
    self.feature_handler = None
    self.model_dir = Path(self.cfg['model_dir'])
    self.label_dict = None

    # some basic checks
    assert self.model_dir.is_dir(), "Your model directory does not exist: {}!".format(self.model_dir)

    # define
    self.define_and_check()


  def cfg_init(self, **cfg_overwrites):
    """
    config init
    """

    # default config
    cfg_default = {
      'add_python_paths': [],
      'target_sample_rate': 24000,
      'target_wav_length_sec': 3,
      'feature_handler': {
        'module': 'feature_handler',
        'attr': 'FeatureHandler',
        'args': [],
        'kwargs': {
          'target_sample_rate': 24000,
          'window_len': 4096,
          'window_stride': 512,
          'window_scaling_bits': 12,
          'mel_n_channels': 40,
          'mel_low_hz': 125,
          'mel_high_hz': 7500,
          'mel_post_scaling_bits': 6,
          'transpose_features_extracted': True,
          'normalize_features': True,
          'to_float': True,
          }
        },
      }

    # config update
    self.cfg = {**cfg_default, **cfg_overwrites, **self.cfg, **self.kwargs}


  def define_and_check(self):
    """
    define and check
    """

    # label dict file in same folder as model
    assert Path(self.cfg['label_dict_yaml_path']).is_file(), "Label dict yaml file does not exist at: {}".format(self.cfg['label_dict_yaml_path'])

    # extract label dict
    self.label_dict = yaml.safe_load(open(self.cfg['label_dict_yaml_path']))['label_dict']

    # info
    print("Using num classes: {} in label dict: {}".format(len(self.label_dict), self.label_dict))

    # classes
    feature_handler_class = getattr(importlib.import_module(self.cfg['feature_handler']['module']), self.cfg['feature_handler']['attr'])
    model_class = getattr(importlib.import_module(self.cfg['model']['module']), self.cfg['model']['attr'])

    # instances
    self.feature_handler = feature_handler_class(*self.cfg['feature_handler']['args'], **self.cfg['feature_handler']['kwargs'])

    # check feature handler
    self.check_feature_handler()

    # determine feature size for model
    feature_sample_test_shape = self.feature_handler.extract(np.random.randn(self.cfg['target_sample_rate'] * self.cfg['target_wav_length_sec'])).shape

    # add channel dimension if required
    if len(feature_sample_test_shape) == 2: feature_sample_test_shape = (1,) + feature_sample_test_shape

    # model kwargs
    model_kwargs = {'input_shape': list(feature_sample_test_shape), 'num_classes': len(self.label_dict), 'is_inference_model': True}

    # model
    self.model = model_class(*self.cfg['feature_handler']['args'], **{**self.cfg['feature_handler']['kwargs'], **model_kwargs})

    # check model
    self.check_model()

    # load model
    model_files = list(self.model_dir.glob('*' + self.cfg['saved_model_extension']))

    # assertions about model file
    assert len(model_files), "No model files found in {} with extension: {}".format(self.model_dir, self.cfg['saved_model_extension'])
    assert len(model_files) == 1, "Sorry only one model allowed in {}, found: {}".format(self.model_dir, len(model_files))

    # get model file
    target_model_file = model_files[0]

    # info
    print("\nModel class: {}\nload model file: [{}]\n".format(self.model.__class__.__name__, target_model_file))

    # load model
    self.model.load(target_model_file)

    # set to evaluation model
    self.model.set_model_to_evaluation_mode()


  def infer(self, wav, fs=24000):
    """
    inference
    """

    # wav check
    self.wav_check(wav, fs)

    # feature extraction
    features = self.feature_handler.extract(wav)

    # to torch
    features = torch.from_numpy(features[np.newaxis, :])

    # model inference
    y_hat = self.model.predict(features)

    return y_hat


  def check_model(self):
    """
    check model - must have a inference function: predict
    """

    # assert existance
    assert not self.model is None, "Model is not initialized (still None)!"

    # method list
    method_list = [m for m in dir(self.model) if callable(getattr(self.model, m)) and not m.startswith('__')]

    # check methods
    assert 'predict' in method_list, "Your model class has no method 'predict'!!!"
    assert 'load' in method_list, "Your model class has no method 'load'!!!"
    assert 'set_model_to_evaluation_mode' in method_list, "Your model class has no method 'set_model_to_evaluation_mode'!!!"


  def check_feature_handler(self):
    """
    check feature handler
    """

    # assert existance
    assert not self.feature_handler is None, "Feature handler is not initialized (still None)!"

    # method list
    method_list = [m for m in dir(self.feature_handler) if callable(getattr(self.feature_handler, m)) and not m.startswith('__')]

    # check methods
    assert 'extract' in method_list, "Your feature handler class has no method 'extract'!!!"


  def wav_check(self, wav, fs):
    """
    check wav
    """

    # assertions
    assert fs == self.cfg['target_sample_rate'], "Wrong fs! Yours: {}, should be: {}".format(fs, self.cfg['target_sample_rate'])
    assert len(wav) == self.cfg['target_wav_length_sec'] * fs, "Length of your audio is wrong! Yours: {}, should be: {}".format(len(wav), self.cfg['target_wav_length_sec'] * fs)


  # --
  # getter

  def get_label_dict(self): return self.label_dict


if __name__ == '__main__':
  """
  inference handler
  """

  # yaml config file
  cfg = yaml.safe_load(open('./config_inference.yaml'))

  # inference handler
  inference_handler = InferenceHandler(cfg['inference_handler'], add_python_paths=[str(Path(__file__).parent.parent)])

  # test audio
  waveform = np.random.randn(24000 * 3)
  fs = 24000

  # infer
  y_hat = inference_handler.infer(waveform, fs)

  # info
  print("predicted: ", y_hat)