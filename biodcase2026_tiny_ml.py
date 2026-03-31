# --
# biodcase 2026 - tiny ml (task 3)

import yaml
import sys
import torch
import numpy as np

from pathlib import Path

from datamodule_tiny_ml import DatamoduleTinyMl
from model_tiny_ml import ModelTinyMl
from plots import plot_confusion_matrix
from compile_embedded_src_code import run_compile_embedded_src_code
from deploy_embedded_compiled_code import run_deploy_embedded_compiled_code
from biodcase_tiny.embedded.esp_target import ESPTarget
from biodcase_tiny.embedded.esp_toolchain import ESPToolchain
from biodcase_tiny.feature_extraction.feature_extraction import make_constants


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


def run_create_target_embedded_src_code(cfg, model_path):
  """
  embedded src code creation
  """

  # check model path
  if not model_path.is_file(): 
    print("\n***Could not create target!!! Your model file does not exist: {}".format(model_path))
    sys.exit()
    return

  # path variables
  template_dir = Path(cfg['generate_embedded_code']['template_dir'])
  gen_code_dir = Path(cfg['generate_embedded_code']['gen_code_dir'])

  # create directory
  if not gen_code_dir.is_dir(): gen_code_dir.mkdir(parents=True)

  # configs
  fe_c = cfg['datamodule']['feature_extraction']
  feature_config = make_constants(sample_rate=cfg['datamodule']['target_sample_rate'], win_samples=fe_c['window_len'], window_scaling_bits=fe_c['window_scaling_bits'], mel_n_channels=fe_c['mel_n_channels'], mel_low_hz=fe_c['mel_low_hz'], mel_high_hz=fe_c['mel_high_hz'], mel_post_scaling_bits=fe_c['mel_post_scaling_bits'])

  # info
  print("\nTarget creation...\n")

  # target creation, validation, and saving of tflite model
  target = ESPTarget(template_dir, model_path, feature_config, reference_dataset_path=None, quantize=cfg['generate_embedded_code']['quantize'])

  # source path
  src_path = gen_code_dir / cfg['generate_embedded_code']['gen_code_source_folder_name']
  src_path.mkdir(exist_ok=True)

  # write templates
  target.process_target_templates(src_path)


if __name__ == '__main__':
  """
  tiny ml starts here
  """

  # yaml config file
  cfg = yaml.safe_load(open('./config.yaml'))

  # info
  print("Hello Tiny ML 2026, version: {}".format(cfg['version']))

  # datamodules
  datamodule_train = DatamoduleTinyMl(cfg['datamodule'], load_set_on_init='train')
  datamodule_validation = DatamoduleTinyMl(cfg['datamodule'], load_set_on_init='validation')
  datamodule_test = DatamoduleTinyMl(cfg['datamodule'], load_set_on_init='test')
  datamodule_train.info()

  # dataloader
  dataloader_train = torch.utils.data.DataLoader(datamodule_train, **cfg['dataloader_train_kwargs'])
  dataloader_validation = torch.utils.data.DataLoader(datamodule_train, **cfg['dataloader_validation_and_test_kwargs'])
  dataloader_test = torch.utils.data.DataLoader(datamodule_train, **cfg['dataloader_validation_and_test_kwargs'])

  # model
  model = ModelTinyMl(cfg['model'], input_shape=datamodule_train.get_feature_shape_at_load(), num_classes=len(datamodule_train.get_label_dict()))

  # run model training
  run_model_training(cfg, model, dataloader_train, dataloader_validation, label_dict=datamodule_test.get_label_dict())

  # run model testing
  run_model_testing(cfg, model, dataloader_test, label_dict=datamodule_test.get_label_dict())

  # run generate embedded src code
  run_create_target_embedded_src_code(cfg, model.get_tflite_model_file_path())

  # compile
  run_compile_embedded_src_code(cfg)

  # deploy
  run_deploy_embedded_compiled_code(cfg)