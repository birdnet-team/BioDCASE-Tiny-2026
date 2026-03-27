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
    self.feature_shape = None
    self.model_dir = Path(self.cfg['model_dir'])
    self.label_dict = None
    self.is_keras_model = None
    self.target_model_file = None

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
      'model_dir': './your_submission_model',
      'saved_model_extension': '.pth',
      'label_dict_yaml_path': './your_submission_model/label_dict.yaml',
      'use_argmax_after_model_predict': False,
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
          'to_torch': True,
          'add_channel_dimension': True,
          'add_batch_dimension': True,
          'channel_dimension_at_end': False,
          }
        },
      'model': {
        'module': 'model_tiny_ml',
        'attr': 'ModelTinyMl',
        'args': [],
        'kwargs': {},
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

    # class
    feature_handler_class = getattr(importlib.import_module(self.cfg['feature_handler']['module']), self.cfg['feature_handler']['attr'])

    # instances
    self.feature_handler = feature_handler_class(*self.cfg['feature_handler']['args'], **self.cfg['feature_handler']['kwargs'])

    # check feature handler
    self.check_feature_handler()

    # determine feature size for model
    self.feature_shape = self.feature_handler.extract(np.random.randn(self.cfg['target_sample_rate'] * self.cfg['target_wav_length_sec'])).shape

    # create model instance
    self.create_and_load_model_instance()


  def create_and_load_model_instance(self):
    """
    create model instance
    """

    # get model files
    model_files = list(self.model_dir.glob('*' + self.cfg['saved_model_extension']))

    # assertions about model file and feature shape
    assert len(model_files), "No model files found in {} with extension: {}".format(self.model_dir, self.cfg['saved_model_extension'])
    assert len(model_files) == 1, "Sorry only one model allowed in {}, found: {}".format(self.model_dir, len(model_files))
    assert self.feature_shape is not None, "This should not happen, something wrong with feature shape!"

    # get model file
    self.target_model_file = model_files[0]

    # keras model?
    self.is_keras_model = self.cfg['model']['module'] == 'keras' and self.cfg['model']['attr'] == 'Model'

    # keras -> simply load the model and everything is set up
    if self.is_keras_model: 
      import keras
      self.model = keras.models.load_model(self.target_model_file)
      return

    # all other models than keras must be checked!!
    # model class
    model_class = getattr(importlib.import_module(self.cfg['model']['module']), self.cfg['model']['attr'])

    # feature shape remove batch dimension
    feature_shape_no_batch_dim = self.feature_shape[1:] if len(self.feature_shape) == 4 else self.feature_shape

    # model kwargs
    model_kwargs = {'input_shape': list(feature_shape_no_batch_dim), 'num_classes': len(self.label_dict), 'is_inference_model': True}

    # model
    self.model = model_class(*self.cfg['model']['args'], **{**self.cfg['model']['kwargs'], **model_kwargs})

    # check model
    self.check_model()

    # load model
    self.model.load(self.target_model_file)

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

    # model inference
    y_hat = self.model.predict(features)

    # do argmax
    if self.cfg['use_argmax_after_model_predict']: y_hat = np.argmax(y_hat, axis=-1)

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


  def info(self):
    """
    info
    """
    print("\n--\nInference handler info:")
    print("model class: {}{}".format(self.model.__class__.__name__, ' is keras model' if self.is_keras_model else ''))
    print("loaded model file: [{}]".format(self.target_model_file))
    print("label dict: {}".format(self.label_dict))
    print("using num classes: {}".format(len(self.label_dict)))
    print("feature shape: {}".format(self.get_feature_shape()))
    print("--\n")


  # --
  # getter

  def get_label_dict(self): return self.label_dict
  def get_feature_shape(self): return self.feature_shape


if __name__ == '__main__':
  """
  inference handler
  """

  # hide warnings of tensorflow
  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

  # yaml config file
  cfg = yaml.safe_load(open('./config.yaml'))

  # inference handler
  inference_handler_pt = InferenceHandler(cfg['inference_handler_pytorch'], add_python_paths=[str(Path(__file__).parent.parent)])
  inference_handler_tf = InferenceHandler(cfg['inference_handler_tensorflow'], add_python_paths=[str(Path(__file__).parent.parent)])

  # test audio
  waveform = np.random.randn(24000 * 3)
  fs = 24000

  # inference handler
  for inference_handler in [inference_handler_pt, inference_handler_tf]:

    # show some info
    inference_handler.info()

    # infer
    y_hat = inference_handler.infer(waveform, fs)

    # info
    print("predicted: ", y_hat)