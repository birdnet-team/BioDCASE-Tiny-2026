import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import docker
from tqdm import tqdm


logger = logging.getLogger("biodcase_tiny")
logger.setLevel(logging.INFO)


def line_accumulator(byte_iterable):
    """
    Wraps a bytes iterable to accumulate bytes until a CRLF (\r\n) sequence is found.
    """
    buffer = bytearray()
    for chunk in byte_iterable:
        if not chunk:
            continue
        buffer.extend(chunk)
        while True:
            crlf_pos = buffer.find(b'\r\n')
            if crlf_pos == -1:  # No CRLF found
                break
            line = buffer[:crlf_pos]
            yield bytes(line)
            buffer = buffer[crlf_pos + 2:]
    if buffer:
        yield bytes(buffer)


class ESP_IDF_v5_2:
    DOCKER_IMAGE_NAME = "espressif/idf:release-v5.4"

    def __init__(self, port):
        self.dc = docker.from_env()
        self.port = port

    def setup(self):
        images = self.dc.images.list(filters={"reference": self.DOCKER_IMAGE_NAME})
        if images:
            return  # we already pulled it

        # TODO: this progress tracking is near useless
        #   but multiple tqdm bars make pycharm console go crazy
        for line in tqdm(
            self.dc.api.pull(f"docker.io/{self.DOCKER_IMAGE_NAME}", stream=True, decode=True),
            desc="Pulling image",
            leave=True,
            file=sys.stdout,
        ):
            pass

    def _create_container(self, command, volumes: Optional[list]=None, **kwargs):
        return self.dc.containers.run(
            image=self.DOCKER_IMAGE_NAME,
            remove=True,
            volumes=[] if volumes is None else volumes,
            working_dir="/project",
            user=os.getuid(),
            environment={"HOME": "/tmp"},
            command=command,
            group_add=["plugdev", "dialout"],
            detach=True,
            **kwargs
        )

    def compile(self, src_path: Path):
        container = self._create_container(
            "idf.py build",
            volumes=[f"{str(src_path)}:/project"],
        )
        try:
            output = container.attach(stdout=True, stream=True, logs=True)
            for line in output:
                logger.info(line.decode("utf-8").rstrip())
            status = container.wait()
            exit_code = status["StatusCode"]
            if exit_code != 0:
                raise ValueError(f"Container exited with exit code {exit_code}")
        except (Exception, KeyboardInterrupt):
            container.stop()
            raise

    def flash(self, src_path: Path):
        container = self._create_container(
            command=f"idf.py -p {self.port} flash",
            volumes=[f"{str(src_path)}:/project"],
            devices=[f"{self.port}:{self.port}"],
        )
        try:
            output = container.attach(stdout=True, stream=True, logs=True)
            for line in output:
                logger.info(line.decode("utf-8").rstrip())
        except (Exception, KeyboardInterrupt):
            container.stop()

    def monitor(self, src_path):
        container = self._create_container(
            command=f"idf.py -p {self.port} monitor",
            volumes=[f"{str(src_path)}:/project"],
            devices=[f"{self.port}:{self.port}"],
            tty=True,
        )
        try:
            output = container.attach(stdout=True, stream=True, logs=True)
            for line in line_accumulator(output):
                print(line.decode("utf-8").rstrip())
        except (Exception, KeyboardInterrupt):
            container.stop()
