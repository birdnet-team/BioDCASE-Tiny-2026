# --
# deploy generated code

import sys

from pathlib import Path
from pipeline_tensorflow.config import Config, load_config
from pipeline_tensorflow.paths import GEN_CODE_DIR

# required package paths
[sys.path.append(p) for p in [str(Path(__file__).parent.parent)] if p not in sys.path]

from biodcase_tiny.embedded.esp_toolchain import ESPToolchain


def run_deploy_generated_code(config: Config, gen_code_dir: Path = GEN_CODE_DIR):
  """
  deploy generated code
  """

  # source path
  src_path = gen_code_dir / "src"

  # assertions
  assert src_path.is_dir(), "Generated code does not exist in {}, run code generation first!".format(gen_code_dir)

  # toolchain: compile, flash, and monitor
  toolchain = ESPToolchain(config.embedded_code_generation.serial_device)
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