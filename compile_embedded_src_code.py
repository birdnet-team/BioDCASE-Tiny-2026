# --
# compile embedded src code

from pathlib import Path
from biodcase_tiny.embedded.esp_toolchain import ESPToolchain
from embedded_code_generation import run_compile_embedded_src_code


if __name__ == '__main__':
  """
  compile embedded src code
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open('./config.yaml'))

  # compile
  run_compile_embedded_src_code(cfg)