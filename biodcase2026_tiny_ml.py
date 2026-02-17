# --
# biodcase 2026 - tiny ml (task 3)

import torch

from datamodule_tiny_ml import DatamoduleTinyMl


if __name__ == '__main__':
  """
  tiny ml starts here
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # info
  print("Hello Tiny ML 2026, version: {}".format(cfg['version']))

  # datamodule
  datamodule = DatamoduleTinyMl(cfg['datamodule'])
  datamodule.info()

  # train dataset
  datamodule.load_train_dataset()

  # loader
  dataloader = torch.utils.data.DataLoader(datamodule, **cfg['dataloader_kwargs'])

  # test loader
  x, y, sid = next(iter(dataloader))

  print("sid: ", sid)