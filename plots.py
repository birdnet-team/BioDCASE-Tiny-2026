# --
# plots

import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable


def rescale_timeticks_iteratively(s, x):
  """
  rescale time ticks iteratively
  """
  assert len(s) == len(x)
  if len(s) <= 20: return s, x
  s, x = rescale_timeticks_iteratively(s[::2] , x[::2])
  return s, x


def add_time_to_axis(ax, num_samples, **kwargs):
  """
  add time to axis
  """

  # fs must be in kwargs
  if 'fs' not in kwargs.keys(): return

  # fs
  fs = kwargs['fs']

  # times
  times = np.arange(0, num_samples) / fs

  # time labels
  s_times = np.arange(0, len(times))
  x_times = ['{:.1f}'.format(ti) for ti in times]

  # rescale timeticks
  s_times, x_times = rescale_timeticks_iteratively(s_times, x_times)

  # set axis
  ax.set_xticks(s_times)
  ax.set_xticklabels(x_times)
  ax.set_xlabel('Time [s]')


def add_frame_time_to_axis(ax, num_samples, **kwargs):
  """
  add time to axis
  """

  # requirements
  #if not kwargs.get('feature_type') in ['spec', 'log_spec', 'log_mel_spec',  'cep', 'mfcc', 'recurrence_matrix']: return
  if 'spec_fs' not in kwargs.keys(): return

  # spec fs
  spec_fs = kwargs['spec_fs']

  # times
  times = np.arange(0, num_samples) / spec_fs

  # time labels
  s_times = np.arange(0, len(times))
  x_times = ['{:.2f}'.format(ti) for ti in times]

  # rescale timeticks
  s_times, x_times = rescale_timeticks_iteratively(s_times, x_times)

  # set axis
  ax.set_xticks(s_times)
  ax.set_xticklabels(x_times)
  ax.set_xlabel('Time [s]')
  

def plot_waveform_and_features(waveform, features, plot_path=None, show_plot_flag=False, **kwargs):
  """
  plot waveform and features
  """

  # create figure
  fig = plt.figure(figsize=(12, 8))

  # adjustments
  fig.subplots_adjust(hspace=0.5)

  # kwargs
  imshow_kwargs = {'origin': 'lower', 'aspect': 'auto', 'interpolation': 'none'}
  [imshow_kwargs.update({k: v}) for k, v in kwargs.items() if k in imshow_kwargs.keys()]

  # title
  if kwargs.get('title') is not None: fig.suptitle(kwargs.get('title'))

  # waveform
  ax = fig.add_subplot(2, 1, 1)
  ax.plot(waveform)
  ax.set_title('Waveform')
  ax.set_ylabel('Magnitude')
  ax.set_xlim([0, len(waveform)])
  add_time_to_axis(ax, len(waveform), **kwargs)
  ax.grid()

  # features
  ax = fig.add_subplot(2, 1, 2)
  im = ax.plot(features) if len(features.shape) == 1 else ax.imshow(features, **imshow_kwargs)
  ax.set_title('Features')
  ax.set_ylabel('Frequency Coeffs.')
  if len(features.shape) == 2: 
    add_frame_time_to_axis(ax, features.shape[1], **kwargs)
    #add_frequency_to_axis(ax, features.shape[0], **kwargs)
  else: 
    add_time_to_axis(ax, features.shape[-1], **kwargs)
    ax.set_xlim([0, features.shape[-1]])
    ax.grid()

  # add colorbar
  add_colorbar(im, fig, cax=None, size='2%', pad='2%')

  # save figure
  if plot_path is not None: fig.savefig(plot_path, dpi=100)

  # show plot or close
  plt.show() if show_plot_flag else plt.close()


def plot_confusion_matrix(y_target, y_predicted, labels=None, plot_path=None, show_plot_flag=False, apply_argmax=False):
  """
  plot confusion matrix
  """

  # argmax targets and predictions
  if apply_argmax:
    y_target = np.argmax(y_target, axis=1)
    y_predicted = np.argmax(y_predicted, axis=1)

  # labels
  labels = labels if not labels is None and not len(np.unique([y_target, y_predicted])) != len(labels) else np.unique([y_target, y_predicted])

  # confusion matrix
  confusion_matrix = sklearn.metrics.confusion_matrix(y_target, y_predicted)

  # max value
  max_value = int(np.max(np.sum(confusion_matrix, axis=1)))

  # create figure
  fig = plt.figure(figsize=(12, 8))

  # create axis
  ax = fig.add_subplot(1, 1, 1)

  # kwargs
  imshow_kwargs = {'origin': 'upper', 'aspect': 'equal', 'interpolation': 'none', 'cmap': 'OrRd', 'vmax': max_value, 'vmin': 0}

  # image
  im = ax.imshow(confusion_matrix, **imshow_kwargs)

  # text handling
  for y_pred_pos in range(len(labels)):
    for y_true_pos in range(len(labels)):

      # font color and size
      font_color = 'black' if confusion_matrix[y_pred_pos, y_true_pos] < 0.6 * max_value else 'white'
      fontsize = 8 if len(labels) > 10 else 11

      # write numbers inside
      ax.text(y_true_pos, y_pred_pos, confusion_matrix[y_pred_pos, y_true_pos], ha='center', va='center', color=font_color, fontsize=fontsize, bbox=None)

  # care about labels
  ax.set_xticks(np.arange(len(labels)))
  ax.set_yticks(np.arange(len(labels)))
  ax.set_xticklabels(labels)
  ax.set_yticklabels(labels)
  ax.set_xlabel('Predicted Labels', fontsize=13)
  ax.set_ylabel('True Labels', fontsize=13)
  ax.set_title('Confusion Matrix with Accuracy: {:.4f}'.format(np.mean(np.array(y_predicted) == np.array(y_target)).item()))
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

  # spacing
  fig.subplots_adjust(bottom=0.25)

  # add colorbar
  add_colorbar(im, fig, cax=None, size='3%', pad='3%')

  # save figure
  if plot_path is not None: fig.savefig(plot_path, dpi=100)

  # show plot or close
  plt.show() if show_plot_flag else plt.close()


def add_colorbar(im, fig, cax=None, size='2%', pad='2%', **kwargs):
  """
  adds colorbar to 2d plot
  """

  # devider for cax
  if cax is None: cax = make_axes_locatable(plt.gca()).append_axes('right', size=size, pad=pad)

  # colorbar
  color_bar = fig.colorbar(im, cax=cax)
  color_bar.ax.tick_params(labelsize=10)
