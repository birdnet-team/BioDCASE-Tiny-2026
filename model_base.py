# --
# model base

import sys
import numpy as np
import torch
import importlib
from pathlib import Path


class ModelBase(torch.nn.Module):
  """
  model base
  """

  def __init__(self, cfg={}, **kwargs):

    # arguments
    self.cfg = cfg
    self.kwargs = kwargs

    # init config
    self.cfg_init()

    # parent init
    super().__init__()

    # add python paths
    [sys.path.append(p) for p in self.cfg['add_python_paths'] if p not in sys.path]

    # members
    self.criterion = None
    self.optimizer = None
    self.device = torch.device(self.cfg['device']['device_name'] if torch.cuda.is_available() and not self.cfg['device']['use_cpu'] else 'cpu')
    self.model_file_path = Path(self.cfg['save_path']) / (self.cfg['model_name'] + '.pth')
    self.add_members()

    # create save path
    if not Path(self.cfg['save_path']).is_dir(): Path(self.cfg['save_path']).mkdir(parents=True)

    # print device
    if self.cfg['verbose']: print("{} on device: {}".format(self.cfg['model_name'], self.device) + ("\nGPU: {}".format(torch.cuda.get_device_name(self.device)) if torch.cuda.is_available() and not self.cfg['device']['use_cpu'] else ""))

    # define
    self.define_network_structure()

    # setup
    self.setup()


  def cfg_init(self, **cfg_overwrites):
    """
    config init
    """

    # default config
    cfg_default = {
      'model_name': self.__class__.__name__,
      'add_python_paths': ['/world/dekutree/git/mypylib/'],
      'save_path': './output/03_model',
      'input_shape': [32],
      'num_classes': 3,
      'prediction_type': 'argmax_classification',
      'prediction_types': ['pass_through', 'argmax_classification'],
      'device': {'use_cpu': False, 'device_name': 'cuda:0'},
      'criterion': {'module': 'torch.nn', 'attr': 'CrossEntropyLoss', 'kwargs': {}},
      'optimizer': {'module': 'torch.optim', 'attr': 'Adam', 'kwargs': {'lr': 0.0001, 'betas': [0.9, 0.999]}},
      'use_early_stopping_criteria': True,
      'verbose': False
    }

    # config update
    self.cfg = {**cfg_default, **cfg_overwrites, **self.cfg, **self.kwargs}


  def add_members(self):
    """
    add members to class
    """
    pass


  def define_network_structure(self):
    """
    define network structure
    """

    # layer
    self.layer = torch.nn.Sequential(
      torch.nn.Linear(torch.zeros(self.cfg['input_shape']).shape.numel(), self.cfg['num_classes']),
      torch.nn.Softmax(dim=1),
      )


  def setup(self):
    """
    setup
    """

    # model to device
    self.to(device=self.device)

    # optimizer and criterion
    self.optimizer = getattr(importlib.import_module(self.cfg['optimizer']['module']), self.cfg['optimizer']['attr'])(self.parameters(), **self.cfg['optimizer']['kwargs'])
    self.criterion = getattr(importlib.import_module(self.cfg['criterion']['module']), self.cfg['criterion']['attr'])(**self.cfg['criterion']['kwargs'])
    

  def forward(self, x):
    """
    forward pass
    """

    # 1. layer
    y = self.layer(x)

    return y


  def train_step(self, data):
    """
    training step
    """

    # reset optimizer
    self.optimizer.zero_grad()

    # get data
    x = data[0].to(device=self.device, dtype=torch.float32)
    y = data[1].to(device=self.device)

    # forward
    y_hat = self.forward(x)

    # loss
    loss = self.criterion(y_hat, y)

    # backwards
    loss.backward()

    # parameter update
    self.optimizer.step()

    return loss.item()


  def validation_step(self, data):
    """
    validation step
    """

    # no gradients here
    with torch.no_grad():

      # get data
      x = data[0].to(device=self.device, dtype=torch.float32)
      y = data[1].to(device=self.device)

      # forward 
      y_hat = self.forward(x)

      # loss
      loss = self.criterion(y_hat, y)

      # prediction - argmax
      y_pred = torch.argmax(y_hat, axis=-1).cpu().numpy()

    return y_pred, loss.item()


  def predict(self, x):
    """
    predict
    """

    # make sure eval mode is activated
    self.eval()

    # no gradients here
    with torch.no_grad():

      # get data
      x = x.to(device=self.device, dtype=torch.float32)

      # forward 
      y_hat = self.forward(x)

      # prediction - argmax
      y_pred = torch.argmax(y_hat, axis=-1).cpu().numpy()

    return y_pred


  def save(self):
    """
    save model
    """
    torch.save(self.state_dict(), self.model_file_path)


  def load(self, model_file):
    """
    load model
    """
    self.load_state_dict(torch.load(model_file, map_location=self.device))


  def count_params(self):
    """
    count all parameters
    """
    return [p.numel() for p in self.parameters() if p.requires_grad]


  def count_operations(self):
    """
    calculate amount of operations, multiplication and add (MAD or MAC) is one operation
    """

    # net dimensions
    net_dim = {'in': self.cfg['input_shape']}
    net_ops = {'in': 0}

    for module in self.modules():

      # conv2d
      if isinstance(module, torch.nn.Conv2d):
        module_name = 'conv{}'.format(len([k for k in net_dim.keys() if 'conv' in k]))
        last_dim = list(net_dim.values())[-1]
        new_dim = [module.out_channels] + [(d - k) // s + 1 for d, k, s in zip(last_dim[1:], module.kernel_size, module.stride)]
        ops = np.prod(new_dim) * np.prod(module.kernel_size) * last_dim[0] + module.out_channels * np.prod(module.kernel_size)
        net_dim.update({module_name: new_dim})
        net_ops.update({module_name: ops.item()})
        continue

      # linear
      if isinstance(module, torch.nn.Linear): 
        module_name = 'lin{}'.format(len([k for k in net_dim.keys() if 'lin' in k]))
        last_dim = list(net_dim.values())[-1]
        new_dim = module.out_features
        ops = np.prod(last_dim) * new_dim
        net_dim.update({module_name: new_dim})
        net_ops.update({module_name: ops.item()})
        continue

    return net_dim, net_ops


  def info(self, num_samples=64,**kwargs):
    """
    info
    """

    import torchinfo
    import subprocess
    import re

    # params and ops
    params = self.count_params()
    net_dim, net_ops = self.count_operations()

    # cpu info
    cpu_info = '' if not str(self.device) == 'cpu' else re.sub(r'Model name:\s*', '', re.findall(r'Model name:.*', (subprocess.check_output("lscpu", shell=True).strip()).decode())[0])

    # show info
    print("\n--\n{} info: ".format(self.cfg['model_name']))
    print("model params: ", [(list(p.shape), p.type()) for p in self.parameters()])
    print("num params: {}, sum: {}".format(params, '{}K'.format(sum(params) / 1000.0) if int(sum(params) / 1000.0) else sum(params)))
    print("net dim: ", net_dim)
    print("net ops: ", net_ops)
    print("total ops: ", sum(net_ops.values()))
    print("device: {}".format(self.device) + (" -- GPU: {}".format(torch.cuda.get_device_name(self.device)) if torch.cuda.is_available() and not self.cfg['device']['use_cpu'] else " -- CPU: {}".format(cpu_info)))
    print("torch info: "), torchinfo.summary(self, (1,) + tuple(self.cfg['input_shape']), col_names=["kernel_size", "output_size", "num_params", "mult_adds"])
    print("--\n")

    # cleanup device because of torchinfo
    self.setup()


  # --
  # getter

  def get_cfg(self): return self.cfg
  def get_input_shape(self): return self.cfg['input_shape']
  def get_save_path(self): return self.cfg['save_path']
  def get_model_file_path(self): return self.model_file_path
  def get_model_name(self): return self.cfg['model_name']


  # --
  # setter

  def set_model_to_training_mode(self): self.train()
  def set_model_to_evaluation_mode(self): self.eval()



if __name__ == '__main__':
  """
  model base
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # num classes
  num_classes = 4

  # test data sample
  x = torch.randn(4, 32)
  y = torch.randint(0, num_classes, size=(4,))
  data = (x, y)

  # model
  model = ModelBase(save_path='./output/03_model', input_shape=x.shape[1:], num_classes=num_classes, verbose=True, device={'use_cpu': True})
  model.info()

  # to train mode
  model.set_model_to_training_mode()

  # train loop
  for epoch in range(5):

    # train model
    loss = model.train_step(data)
    print("Epoch {:03} with loss: {:6f}".format(epoch + 1, loss))

  # evaluation mode (must be done to disable e.g. dropout)
  model.set_model_to_evaluation_mode()

  # validation step
  y_pred, loss = model.validation_step(data)

  print("actual: ", y)
  print("prediction: ", y_pred)
  print("loss: ", loss)
  print("acc: ", torch.sum(y == y_pred).item() / len(y))

  # save model
  model.save()
