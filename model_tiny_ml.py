# --
# model tiny ml

import numpy as np
import torch

from model_base import ModelBase


class ModelTinyMl(ModelBase):
  """
  model tiny ml - overwrite model base
  """

  def define_network_structure(self):
    """
    define network structure
    """

    # check input shape before, must be [channels, rows, cols]
    assert len(self.cfg['input_shape']) == 3

    # conv layer 1
    self.layer1 = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=self.cfg['input_shape'][0], out_channels=16, kernel_size=(16, 16), stride=(1, 1)),
      torch.nn.ReLU(),
      )
    
    # conv layer 2
    self.layer2 = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(8, 8), stride=(1, 1)),
      torch.nn.Dropout2d(p=0.25),
      torch.nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5)),
      torch.nn.ReLU(),
      )

    # get shape of conv layers
    with torch.no_grad(): flattened_shape = self.layer2(self.layer1(torch.randn((1,) + tuple(self.cfg['input_shape'])))).data.shape

    # output layer
    self.output_layer = torch.nn.Sequential(
      torch.nn.Flatten(),
      torch.nn.Linear(flattened_shape.numel(), flattened_shape.numel()//4),
      torch.nn.Dropout(p=0.5),
      torch.nn.ReLU(),
      torch.nn.Linear(flattened_shape.numel()//4, self.cfg['num_classes']),
      torch.nn.Softmax(dim=1),
      )


  def forward(self, x):
    """
    forward pass
    """

    # 1. conv layer
    h = self.layer1(x)

    # 2. conv layer
    h = self.layer2(h)

    # output layer
    y = self.output_layer(h)

    return y



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
  model = ModelTinyMl(cfg['model'], input_shape=tuple(x.shape[1:]), num_classes=num_classes)
  model.info()

  # data structure
  data = (x, y)

  # to train mode
  model.set_model_to_training_mode()

  # train loop
  for epoch in range(5):

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
