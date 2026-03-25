# --
# feature handler

import numpy as np
import functools

from biodcase_tiny.feature_extraction.feature_extraction import process_window, make_constants


class FeatureHandler():
  """
  feature handler
  """

  def __init__(self, cfg={}, **kwargs):

    # super constructor
    super().__init__()

    # arguments
    self.cfg = cfg
    self.kwargs = kwargs

    # init config
    self.cfg_init()

    # members
    self.feature_constants = None
    self.do_windows_fn = None

    # do definition
    self.define()


  def cfg_init(self, **cfg_overwrites):
    """
    config init
    """

    # default config
    cfg_default = {
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

    # config update
    self.cfg = {**cfg_default, **cfg_overwrites, **self.cfg, **self.kwargs}


  def define(self):
    """
    define
    """

    # feature constants
    self.feature_constants = make_constants(
      win_samples=self.cfg['window_len'],
      sample_rate=self.cfg['target_sample_rate'],
      window_scaling_bits=self.cfg['window_scaling_bits'],
      mel_n_channels=self.cfg['mel_n_channels'],
      mel_low_hz=self.cfg['mel_low_hz'],
      mel_high_hz=self.cfg['mel_high_hz'],
      mel_post_scaling_bits=self.cfg['mel_post_scaling_bits'],
    )

    # window function
    apply_windowed = lambda data, window_len, window_stride, fn: np.array([fn(row) for row in np.lib.stride_tricks.sliding_window_view(data, window_len)[::window_stride]])

    # this partial stuff is just a way to set all config parameters, so we have a function that only takes data as input
    self.do_windows_fn = functools.partial(
      apply_windowed,
      window_len=self.cfg['window_len'],
      window_stride=self.cfg['window_stride'],
      fn=functools.partial(
        process_window,
        hanning=self.feature_constants.hanning_window,
        mel_constants=self.feature_constants.mel_constants,
        fft_twiddle=self.feature_constants.fft_twiddle,
        window_scaling_bits=self.feature_constants.window_scaling_bits,
        mel_post_scaling_bits=self.feature_constants.mel_post_scaling_bits
      ),
    )


  def extract(self, x):
    """
    extract features
    """

    # to numpy
    if not isinstance(x, np.ndarray): x = np.array(x)

    # ensure integer - required for our tiny ml feature extractions
    if np.issubdtype(x.dtype, np.floating):

      # normalize [-1, 1]
      x = x / np.max(np.abs(x))

      # to int16 conversion for serialization
      x = (x * np.iinfo(np.int16).max).astype(np.int16)

    # extract
    x_t = self.do_windows_fn(x)

    # to float?
    if self.cfg['to_float']: x_t = x_t.astype(np.float32)

    # transpose
    if self.cfg['transpose_features_extracted']: x_t = x_t.T

    # normalize [0, 1]
    if self.cfg['normalize_features']: x_t = (x_t - np.min(x_t)) / np.ptp(x_t)

    return x_t



if __name__ == '__main__':
  """
  feature handler
  """

  from plots import plot_waveform_and_features

  # test audio
  waveform = np.random.randn(4096 * 2)

  # feature handler
  feature_handler = FeatureHandler()

  # extract
  features = feature_handler.extract(waveform)
  
  # info  
  print("shapes, x: {}, x_t: {}".format(waveform.shape, features.shape))

  # plot waveform and features
  plot_waveform_and_features(waveform, features, show_plot_flag=True, title='random sample')
