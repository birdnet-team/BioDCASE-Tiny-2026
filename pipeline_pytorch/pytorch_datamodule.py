# --
# datamodule tiny ml

import sys
import re
import torch
import yaml
import numpy as np
import soundfile
import gzip
import functools

from pathlib import Path
from datamodule import DatamoduleTinyMl


class DataloaderPytorch(torch.utils.data.Dataset):
  """
  Pytorch Dataset loader wrapped around the DatamoduleTinyMl
  """

  def __init__(self, datamodule):

    # super constructor
    super().__init__()

    self.datamodule = datamodule


    # Methods to expose from datamodule
    self._EXPOSED = {
        "get_label_dict",
        "get_target_to_label_dict",
        "get_cache_info",
        "get_targets",
        "get_feature_shape_at_load",

    }

  def __len__(self):
    return self.datamodule.length

  def __getitem__(self, idx):
    return torch.from_numpy(self.datamodule.features[idx]), torch.asarray(self.datamodule.targets[idx]), torch.asarray(self.datamodule.sample_ids[idx])
  
  # Exposed methods (getters)
  
  def __getattr__(self, name):
    if name in self._EXPOSED:
        return getattr(self.datamodule, name)
    raise AttributeError(f"{name} not exposed")

  def info(self):
    """
    info
    """
    print("\n--\n{} info: ".format(self.__class__.__name__))
    print("label dict: ", self.datamodule.get_label_dict())
    if self.datamodule.load_info.get('feature_shape_at_load') is not None: print("feature size at load: ", self.datamodule.get_feature_shape_at_load())
    print("--\n")