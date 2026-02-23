# --
# plots

import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_confusion_matrix(y_target, y_predicted, labels=None, plot_path=None, is_showing_plot=False, apply_argmax=False):
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
  ax.set_title('Confusion Matrix')
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

  # spacings
  fig.subplots_adjust(bottom=0.25)

  # divider for cax
  cax = make_axes_locatable(plt.gca()).append_axes('right', size='3%', pad='3%')

  # colorbar
  color_bar = fig.colorbar(im, cax=cax)
  color_bar.ax.tick_params(labelsize=10)

  # save figure
  if plot_path is not None: fig.savefig(plot_path, dpi=100)

  # show plot or close
  plt.show() if is_showing_plot else plt.close()