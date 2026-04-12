# --
# deploy embedded compiled code

from pathlib import Path
from biodcase_tiny.embedded.esp_toolchain import ESPToolchain
from embedded_code_generation import run_deploy_embedded_compiled_code


if __name__ == '__main__':
  """
  deploy embedded compiled code
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open('./config.yaml'))

  # deploy
  run_deploy_embedded_compiled_code(cfg)