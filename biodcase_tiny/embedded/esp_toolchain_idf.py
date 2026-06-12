import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .esp_monitor_parser import parse_monitor_output


logger = logging.getLogger("biodcase_tiny")
logger.setLevel(logging.INFO)


class ESPToolchain:
    """
    ESP-IDF toolchain (native installation).

    This implementation allows your application to run inside its own
    Python virtual environment while ESP-IDF uses its own Python
    environment configured by export.sh.
    """

    def __init__(
        self,
        port: str,
        idf_path: str | Path = "~/.espressif/v5.5.2/esp-idf/",
    ):
        self.port = port
        self.idf_path = Path(idf_path).expanduser().resolve()
        self.idf_env: Optional[dict[str, str]] = None

    def setup(self):
        """
        Load the ESP-IDF environment by sourcing export.sh.
        """

        export_script = self.idf_path / "export.sh"

        if not export_script.exists():
            raise RuntimeError(
                f"ESP-IDF export script not found: {export_script}"
            )

        cmd = f"source '{export_script}' >/dev/null && env"

        try:
            result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to source ESP-IDF export.sh:\n{exc.stderr}"
            ) from exc

        env = os.environ.copy()

        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                env[key] = value

        if shutil.which("idf.py", path=env.get("PATH")) is None:
            raise RuntimeError(
                "idf.py not found after sourcing export.sh.\n"
                f"ESP-IDF path: {self.idf_path}"
            )

        self.idf_env = env

        logger.info(
            "ESP-IDF environment loaded successfully."
        )

    def _run(
        self,
        args: list[str],
        cwd: Path,
        collect_output: bool = False,
    ):
        """
        Execute idf.py commands.
        """

        if self.idf_env is None:
            raise RuntimeError(
                "ESP-IDF environment not initialized. "
                "Call setup() first."
            )

        process = subprocess.Popen(
            args,
            cwd=str(cwd),
            env=self.idf_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        collected = []

        try:
            assert process.stdout is not None

            for line in process.stdout:
                line = line.rstrip()

                logger.info(line)

                if collect_output:
                    collected.append(line)

            exit_code = process.wait()

            if exit_code != 0:
                raise ValueError(
                    f"Command exited with exit code {exit_code}"
                )

            return collected

        except (Exception, KeyboardInterrupt):
            process.terminate()

            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

            raise

    def set_target(
        self,
        src_path: Path,
        target: str = "esp32-s3",
    ):
        """
        Configure ESP-IDF target.
        """

        assert target in ["esp32-s3"], (
            f"Sorry we do not support your selected target: [{target}]"
        )

        src_path = src_path.resolve()

        self._run(
            ["idf.py", "set-target", target],
            cwd=src_path,
        )

    def compile(self, src_path: Path):
        """
        Build project.
        """

        src_path = src_path.resolve()

        self._run(
            ["idf.py", "build"],
            cwd=src_path,
        )

    def flash(self, src_path: Path):
        """
        Flash firmware to the ESP device.
        """

        src_path = src_path.resolve()

        self._run(
            [
                "idf.py",
                "-p",
                self.port,
                "flash",
            ],
            cwd=src_path,
        )

    def monitor(
        self,
        src_path: Path,
        report_file_path: Optional[Path] = None,
    ):
        """
        Monitor ESP output until app_main exits.
        """

        if self.idf_env is None:
            raise RuntimeError(
                "ESP-IDF environment not initialized. "
                "Call setup() first."
            )

        src_path = src_path.resolve()

        process = subprocess.Popen(
            [
                "idf.py",
                "-p",
                self.port,
                "monitor",
            ],
            cwd=str(src_path),
            env=self.idf_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        collected_lines = []

        try:
            assert process.stdout is not None

            for line in process.stdout:
                decoded = line.rstrip()

                print(decoded)

                collected_lines.append(decoded)

                if "main_task: Returned from app_main()" in decoded:
                    parse_monitor_output(
                        collected_lines,
                        report_file_path=report_file_path,
                    )

                    process.terminate()
                    break

            process.wait()

        except (Exception, KeyboardInterrupt):
            if collected_lines:
                parse_monitor_output(
                    collected_lines,
                    report_file_path=report_file_path,
                )

            process.terminate()

            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

            raise