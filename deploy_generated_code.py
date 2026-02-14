# --
# deploy generated code

from config import Config, load_config
from biodcase_tiny.embedded.esp_toolchain import ESP_IDF_v5_2
from paths import GEN_CODE_DIR
from pathlib import Path


def run_deploy_generated_code(config: Config, gen_code_dir: Path = GEN_CODE_DIR):
  """
  deploy generated code
  """

  # assertions
  assert gen_code_dir.is_dir(), "Generated code does not exist in {}, run code generation first!".format(gen_code_dir)

  # source path
  src_path = gen_code_dir / "src"
  src_path.mkdir(exist_ok=True)

  # toolchain: compile, flash, and monitor
  toolchain = ESP_IDF_v5_2(config.embedded_code_generation.serial_device)
  #toolchain.compile(src_path=src_path)
  toolchain.flash(src_path=src_path)
  toolchain.monitor(src_path=src_path)



if __name__ == '__main__':
  """
  deploy generated code
  """

  # config
  config = load_config()

  # run embedded code generation
  run_deploy_generated_code(config)