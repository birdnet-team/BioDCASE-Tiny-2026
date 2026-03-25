# --
# feature extraction

import sys

import json
import faulthandler
import yaml
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow
import matplotlib.pyplot as plt

from pathlib import Path
from functools import partial
from numpy.lib.stride_tricks import sliding_window_view
from tqdm.dask import TqdmCallback

from config import load_config, Config
from paths import PREPROC_PRQ_PATH, FEATURES_PRQ_PATH, FEATURES_SAMPLE_PLOT_DIR, FEATURES_SHAPE_JSON_PATH

# required package paths
[sys.path.append(p) for p in [str(Path(__file__).parent.parent)] if p not in sys.path]

from biodcase_tiny.feature_extraction.feature_extraction import process_window, make_constants


def plot_features_sample(sample: pd.DataFrame, features_shape, plot_path=None, **kwargs):
  """
  Plot a few samples of features
  """

  # create figure
  fig = plt.figure(figsize=(12, 8))

  # adjustments
  fig.subplots_adjust(hspace=0.5, wspace=0.3)

  # rows and cols
  num_rows = len(sample)
  num_cols = 2

  # kwargs
  valid_gs_kwargs = {'width_ratios': None, 'height_ratios': None}
  [valid_gs_kwargs.update({k: v}) for k, v in kwargs.items() if k in valid_gs_kwargs.keys()]

  # image kwargs
  imshow_kwargs = {'aspect': 'auto', 'interpolation': 'none', 'origin': 'lower',}

  # grid spec
  gs = fig.add_gridspec(num_rows, num_cols, **valid_gs_kwargs)

  # add traces
  for i, (idx, sample_row) in enumerate(sample.iterrows()):

    # create axis
    ax_wav = fig.add_subplot(gs[(i*2)])
    ax_fea = fig.add_subplot(gs[(i*2)+1])

    # sample extraction of waveform and features
    waveform, features = sample_row["data"], sample_row["features"].reshape(features_shape).T

    # image
    im_wav = ax_wav.plot(waveform)
    im_fea = ax_fea.imshow(features, **imshow_kwargs)

    # axis settings
    ax_wav.set_title(Path(sample_row['path']).name)
    ax_wav.set_xlabel('Time [samples]')
    ax_wav.set_ylabel('Magnitude')
    ax_wav.grid()
    ax_fea.set_xlabel('Time [frames]')
    ax_fea.set_ylabel('Magnitude')

  # save figure
  if plot_path is not None: fig.savefig(plot_path, dpi=100)

  # close figure
  plt.close()


def run_feature_extraction(config: Config, preproc_prq_path=PREPROC_PRQ_PATH, features_prq_path=FEATURES_PRQ_PATH, features_sample_plot_dir=FEATURES_SAMPLE_PLOT_DIR, features_shape_json_path=FEATURES_SHAPE_JSON_PATH,):
  """
  ferature extraction
  """

  # assertions
  assert preproc_prq_path.is_dir(), "preprocessing directory does not exist: [{}], run preprocessing before!".format(preproc_prq_path)

  # create directory
  if not features_prq_path.parent.is_dir(): features_prq_path.parent.mkdir(parents=True)
  if not features_sample_plot_dir.is_dir(): features_sample_plot_dir.mkdir(parents=True)

  # fault handler?
  faulthandler.enable()

  # dask config
  dask.config.set({"dataframe.convert-string": False})

  # data extraction
  data = dd.read_parquet(preproc_prq_path)

  # read and save yaml of class dict also to feature extraction
  yaml.dump(yaml.safe_load(open(preproc_prq_path.parent / 'label_dict.yaml')), open(features_prq_path.parent / 'label_dict.yaml', 'w'))

  # feature extraction config
  fe_config = config.feature_extraction
  dp_config = config.data_preprocessing
  fc = make_constants(
    fe_config.window_len,
    dp_config.sample_rate,
    fe_config.window_scaling_bits,
    fe_config.mel_n_channels,
    fe_config.mel_low_hz,
    fe_config.mel_high_hz,
    fe_config.mel_post_scaling_bits
  )

  # window function
  apply_windowed = lambda data, window_len, window_stride, fn: np.array([fn(row) for row in sliding_window_view(data, window_len)[::window_stride]])

  # this partial stuff is just a way to set all config parameters, so we have a function that only takes data as input
  do_windows_fn = partial(
    apply_windowed,
    fn=partial(
      process_window,
      hanning=fc.hanning_window,
      mel_constants=fc.mel_constants,
      fft_twiddle=fc.fft_twiddle,
      window_scaling_bits=fc.window_scaling_bits,
      mel_post_scaling_bits=fc.mel_post_scaling_bits
    ),
    window_len=fe_config.window_len,
    window_stride=fe_config.window_stride,
  )

  # feature shape of example
  features_shape = do_windows_fn(data["data"].head(1)[0]).shape

  # extract partitions
  with TqdmCallback(desc="Extracting features from preprocessed data"):

    # data processing
    data = data.map_partitions(
      lambda df: df.assign(features=df["data"].apply(lambda clip: do_windows_fn(clip).flatten(),)),
      meta=pd.DataFrame(dict(**{c: data._meta[c] for c in data._meta}, features=pd.Series([], dtype=float),))
      )

    # todo: create plots of each class
    # choose a sample and plot it
    plot_features_sample(data.head(3), features_shape, plot_path=features_sample_plot_dir / 'features_sample.png')

    # remove original audio
    data: dd.DataFrame = data.drop("data", axis=1)

    # add features to parquet, make sure array is serialized correctly
    data.to_parquet(
      features_prq_path,
      schema={'features': pyarrow.list_(pyarrow.float32())},
      write_index=False,
    )

  # save the feature shape as rows are flattened to recover them later
  json.dump(features_shape, open(features_shape_json_path, 'w'))


if __name__ == "__main__":
  """
  feature extraction
  """

  # config
  config = load_config()

  # run feature extraction
  run_feature_extraction(config)
