# --
# compile embedded src code

from pathlib import Path
from biodcase_tiny.embedded.esp_toolchain import ESPToolchain


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


if __name__ == '__main__':
  """
  compile embedded src code
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open('./config.yaml'))

  # compile
  run_compile_embedded_src_code(cfg)