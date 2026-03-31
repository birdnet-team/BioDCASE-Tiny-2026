import yaml
import torch
import numpy as np
from pathlib import Path
from plots import plot_confusion_matrix
from pipeline_pytorch.datamodule_tiny_ml import DatamoduleTinyMl
from pipeline_pytorch.model_tiny_ml import Baseline

def run_model_training(cfg, model, dataloader_train, dataloader_validation, label_dict):
  """
  run model training
  """

  # info
  print("\nTrain model...\n")

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

    # validation loader
    for data in dataloader_validation: 

      # validation step
      y_pred, loss = model.validation_step(data)

      # loss update
      epoch_validation_loss.append(loss)

    # epoch info
    print("Epoch {:03} with train loss: {:.6f} and val loss: {:.6f}".format(epoch + 1, np.mean(epoch_train_loss), np.mean(epoch_validation_loss),))

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
  print("\nTest model...\n")

  # predictions and targets
  y_predictions = []
  y_targets = []

  # test loader
  for data in dataloader_test: 

    # validation step
    y_pred = model.predict(data[0])

    # add data
    y_predictions.extend(y_pred.tolist())
    y_targets.extend(data[1].tolist())

  # accuracy
  acc = np.mean(np.array(y_predictions) == np.array(y_targets)).item()

  # report path
  report_path = Path(cfg['reports']['report_path'])
  if not report_path.is_dir(): report_path.mkdir(parents=True)
  plot_path_cm = report_path / 'cm_{}.png'.format(model.get_model_name())

  # confusion matrix
  plot_confusion_matrix(y_targets, y_predictions, labels=list(label_dict.keys()), plot_path=plot_path_cm)

  # test info
  print("Test accuracy: {:.4f}".format(acc))
  print("Confusion matrix is saved in [{}]".format(plot_path_cm))

  # info
  print("Testing of model finished!")



def pytorch_model_taining(cfg):
  datamodule_train = DatamoduleTinyMl(cfg['datamodule'], load_set_on_init='train')
  datamodule_validation = DatamoduleTinyMl(cfg['datamodule'], load_set_on_init='validation')
  datamodule_test = DatamoduleTinyMl(cfg['datamodule'], load_set_on_init='test')
  datamodule_train.info()

  # dataloader
  dataloader_train = torch.utils.data.DataLoader(datamodule_train, **cfg['dataloader_train_kwargs'])
  dataloader_validation = torch.utils.data.DataLoader(datamodule_train, **cfg['dataloader_validation_and_test_kwargs'])
  dataloader_test = torch.utils.data.DataLoader(datamodule_train, **cfg['dataloader_validation_and_test_kwargs'])

  # model
  model = Baseline(cfg['model'], input_shape=datamodule_train.get_feature_shape_at_load(), num_classes=len(datamodule_train.get_label_dict()))

  # run model training
  run_model_training(cfg, model, dataloader_train, dataloader_validation, label_dict=datamodule_test.get_label_dict())

  # run model testing
  run_model_testing(cfg, model, dataloader_test, label_dict=datamodule_test.get_label_dict())