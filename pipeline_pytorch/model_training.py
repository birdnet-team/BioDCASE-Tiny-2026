# --
# model training

import sys
import yaml
import torch
import numpy as np
import importlib

from torchsummary import summary
from pathlib import Path

# add root path of project if called as main
if __name__ == '__main__': [sys.path.append(p) for p in [str(Path(__file__).parent.parent)] if p not in sys.path]

from plots import plot_confusion_matrix
from pipeline_pytorch.paths import MODELS_DIR, CM_FIG_PATH
from pipeline_pytorch.pytorch_datamodule import DataloaderPytorch
from pipeline_pytorch.model_tiny_ml import Baseline


def run_model_training(cfg, model, dataloader_train, dataloader_validation, label_dict):
  """
  run model training
  """

  # info
  print("\nTrain model on device: {}...\n".format(model.get_device_full_str()))

  # epochs
  for epoch in range(cfg['model_training']['num_epochs']):

    # set to train mode
    model.set_model_to_training_mode()

    # epoch loss
    epoch_train_loss = []
    epoch_validation_loss = []

    # train loader
    for data in dataloader_train: 

      # trainign step
      loss = model.train_step(data)

      # loss update
      epoch_train_loss.append(loss)

    # evaluation mode
    model.set_model_to_evaluation_mode()

    # targets and predictions
    y_targets = np.empty(shape=0, dtype=np.int8)
    y_predictions = np.empty(shape=0, dtype=np.int8)

    # validation loader
    for data in dataloader_validation: 

      # validation step
      y_pred, loss = model.validation_step(data)

      # argmax for acc
      y_pred = np.argmax(y_pred, axis=-1)

      # append targets and predictions
      y_targets = np.append(y_targets, data[1].numpy().astype(np.int8))
      y_predictions = np.append(y_predictions, y_pred.astype(np.int8))

      # loss update
      epoch_validation_loss.append(loss)

    # epoch info
    print("Epoch {:03} - train loss: {:.4f}, val: [loss: {:.4f}, acc: {:.4f}]".format(epoch + 1, np.mean(epoch_train_loss), np.mean(epoch_validation_loss), np.mean(y_targets == y_predictions)))

  # info
  print("Training of model finished!")

  # save model in case of no early stopping
  model.save(save_also_as_tflite=True)

  # save also label dict
  yaml.dump({'label_dict': label_dict}, open(Path(model.get_save_path()) / 'label_dict.yaml', 'w'))


def run_model_testing(cfg, model, dataloader_test, label_dict):
  """
  run model training
  """

  # info
  print("\nTest model on device: {}...\n".format(model.get_device_full_str()))

  # predictions and targets
  y_predictions = []
  y_targets = []

  # test loader
  for data in dataloader_test: 

    # validation step
    y_pred = model.predict(data[0])

    # argmax for acc
    y_pred = np.argmax(y_pred, axis=-1)

    # add data
    y_predictions.extend(y_pred.tolist())
    y_targets.extend(data[1].tolist())

  # accuracy
  acc = np.mean(np.array(y_predictions) == np.array(y_targets)).item()

  # report path
  plot_path_cm = CM_FIG_PATH.parent / 'cm_{}.png'.format(model.get_model_name())

  # confusion matrix
  plot_confusion_matrix(y_targets, y_predictions, labels=list(label_dict.keys()), plot_path=plot_path_cm)

  # test info
  print("Test accuracy: {:.4f}".format(acc))
  print("Confusion matrix is saved in [{}]".format(plot_path_cm))

  # info
  print("Testing of model finished!\n")


def pytorch_model_taining(cfg_framework, datamodule_train, datamodule_validation, datamodule_test):
  """
  pytorch model training
  """
  
  # dataloader
  dataloader_train = torch.utils.data.DataLoader(DataloaderPytorch(datamodule_train), **cfg_framework['dataloader_train_kwargs'])
  dataloader_validation = torch.utils.data.DataLoader(DataloaderPytorch(datamodule_validation), **cfg_framework['dataloader_validation_and_test_kwargs'])
  dataloader_test = torch.utils.data.DataLoader(DataloaderPytorch(datamodule_test), **cfg_framework['dataloader_validation_and_test_kwargs'])

  # model
  input_shape = datamodule_train.get_feature_shape_at_load()
  
  # model class
  model_class = getattr(importlib.import_module(cfg_framework['model']['module']), cfg_framework['model']['attr'])

  # model kwargs
  model_kwargs_overwrite = {'input_shape': input_shape, 'num_classes': len(datamodule_train.get_label_dict()), 'save_path': str(MODELS_DIR)}

  # model
  model = model_class(*cfg_framework['model']['args'], **{**cfg_framework['model']['kwargs'], **model_kwargs_overwrite})

  # summary
  summary(model, input_size=input_shape, device=model.get_device_type_str())

  # run model training
  run_model_training(cfg_framework, model, dataloader_train, dataloader_validation, label_dict=datamodule_train.get_label_dict())

  # run model testing
  run_model_testing(cfg_framework, model, dataloader_test, label_dict=datamodule_test.get_label_dict())

  return model