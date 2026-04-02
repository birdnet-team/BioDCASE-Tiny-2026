# --
# model tiny ml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline_pytorch.model_base import ModelBase


class Baseline(ModelBase):
  """
  model tiny ml - overwrite model base
  """

  def define_network_structure(self, n_filters = 32, dropout=0.05):

    assert len(self.cfg['input_shape']) == 3

    # Feature extractor
    self.features = nn.Sequential(
        nn.Conv2d(self.cfg['input_shape'][0], n_filters, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(n_filters, n_filters * 2, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(4),

        nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3),
        nn.ReLU(),

        nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
    )

    # Classifier
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(n_filters * 4, 32),
        nn.ReLU(),
        nn.Linear(32, self.cfg['num_classes'])
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x



if __name__ == '__main__':
  """
  model tiny ml
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # params
  num_classes = 11
  num_samples = 4

  # test data sample
  y = torch.randint(0, num_classes-1, (num_samples,))

  # model
  x = torch.randn(num_samples, 1, 133, 40)
  model = Baseline(cfg['model'], input_shape=tuple(x.shape[1:]), num_classes=num_classes)
  model.info()

  # data structure
  data = (x, y)

  # to train mode
  model.set_model_to_training_mode()

  # train loop
  for epoch in range(50):

    # train model
    loss = model.train_step(data)
    print("Epoch {:03} with loss: {:6f}".format(epoch + 1, loss))

  # eval mode (must be done to disable for instance dropout)
  model.set_model_to_evaluation_mode()

  # validation step
  y_pred, loss = model.validation_step(data)

  print("actual: ", y)
  print("prediction: ", y_pred)
  print("loss: ", loss)
  print("acc: ", torch.sum(y == y_pred).item() / len(y))

  # save model
  model.save()
