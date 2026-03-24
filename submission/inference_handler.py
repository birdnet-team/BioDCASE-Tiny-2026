# --
# inference handler

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

    # define
    self.define()

    # checks
    self.check_model()
    self.check_feature_handler()


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


  def define(self):
    """
    define
    """

    # classes
    feature_handler_class = getattr(importlib.import_module(self.cfg['feature_handler']['module']), self.cfg['feature_handler']['attr'])
    model_class = getattr(importlib.import_module(self.cfg['model']['module']), self.cfg['model']['attr'])

    # instances
    self.feature_handler = feature_handler_class(*self.cfg['feature_handler']['args'], **self.cfg['feature_handler']['kwargs'])

    # determine feature size for model
    feature_sample_test_shape = self.feature_handler.extract(np.random.randn(self.cfg['target_sample_rate'] * self.cfg['target_wav_length_sec'])).shape

    # add channel dimension if required
    if len(feature_sample_test_shape) == 2: feature_sample_test_shape = (1,) + feature_sample_test_shape

    # model kwargs
    model_kwargs = {'input_shape': list(feature_sample_test_shape), 'num_classes': 11}

    # model
    self.model = model_class(*self.cfg['feature_handler']['args'], **{**self.cfg['feature_handler']['kwargs'], **model_kwargs})

    # todo:
    # load model


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

    print("predict: ", y_hat)
    
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



if __name__ == '__main__':
  """
  inference handler
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("./config_inference.yaml"))

  # inference handler
  inference_handler = InferenceHandler(cfg['inference_handler'], add_python_paths=[str(Path(__file__).parent.parent)])

  # test audio
  waveform = np.random.randn(24000 * 3)
  fs = 24000

  # infer
  inference_handler.infer(waveform, fs)