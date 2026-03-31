import sys
from pathlib import Path

from biodcase_tiny.embedded.esp_target import ESPTarget
from biodcase_tiny.embedded.esp_toolchain import ESPToolchain
from biodcase_tiny.feature_extraction.feature_extraction import make_constants


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


def run_compile_embedded_src_code(cfg):

  # info
  print("\nCode Compilation...\n")

  # source path
  src_path = Path(cfg['generate_embedded_code']['gen_code_dir']) / cfg['generate_embedded_code']['gen_code_source_folder_name']

  # assertions
  assert src_path.is_dir(), "Generated code does not exist in {}, run code creation first!".format(src_path)

  # toolchain: compile, flash, and monitor
  toolchain = ESPToolchain(cfg['generate_embedded_code']['serial_device'])
  #toolchain.set_target(src_path=src_path)
  toolchain.compile(src_path=src_path)


def run_deploy_embedded_compiled_code(cfg):

  # info
  print("\nDeploy code to microcontroller and monitor...")

  # source path
  src_path = Path(cfg['generate_embedded_code']['gen_code_dir']) / cfg['generate_embedded_code']['gen_code_source_folder_name']

  # assertions
  assert src_path.is_dir(), "Generated code does not exist in {}, run code creation first!".format(src_path)

  # toolchain: flash, and monitor
  toolchain = ESPToolchain(cfg['generate_embedded_code']['serial_device'])
  toolchain.flash(src_path=src_path)
  toolchain.monitor(src_path=src_path)