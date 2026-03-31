# --
# deploy embedded compiled code

from pathlib import Path
from biodcase_tiny.embedded.esp_toolchain import ESPToolchain


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


if __name__ == '__main__':
  """
  deploy embedded compiled code
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open('./config.yaml'))

  # deploy
  run_deploy_embedded_compiled_code(cfg)