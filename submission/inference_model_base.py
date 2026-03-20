# --
# inference model base class

import numpy as np


class InferenceModelBase():
  """
  inference model base - use this template for your submission model
  """

  def __init__(self, model, feature_extraction_function, target_fs=24000, target_wav_length_sec=3):
    
    # argmuents
    self.model = model
    self.feature_extraction_function = feature_extraction_function
    self.target_fs = target_fs
    self.target_wav_length_sec = target_wav_length_sec

    # checks
    self.check_model()
    self.check_feature_extraction_function()


  def infer(self, wav, fs=24000):
    """
    inference
    """

    # wav check
    self.wav_check(wav)

    # infer on wav audio
    predicted = self.infer_wav_audio(wav)

    return predicted


  def check_model(self):
    """
    check model - must have a inference function: predict
    """
    pass


  def check_feature_extraction_function(self):
    """
    check feature extraction function
    """
    pass


  def infer_wav_audio(self, wav, fs=24000):
    """
    infer wav audio - implement this
    """
    raise NotImplementedError()


  def wav_check(self, wav, fs):
    """
    check wav
    """

    # assertions
    assert fs == 24000, "Wrong fs! Yours: {}, should be: {}".format(fs)
    assert len(wav) == self.target_wav_length_sec * fs, "Length of your audio is wrong! Yours: {}, should be: {}".format(len(wav), self.target_wav_length_sec * fs)
