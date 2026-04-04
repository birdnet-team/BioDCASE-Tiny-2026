# --
# esp toolchain

import logging
import os
import shutil
import sys
import docker

from pathlib import Path
from typing import Optional
from tqdm import tqdm
from .esp_monitor_parser import parse_monitor_output

# logger
logger = logging.getLogger("biodcase_tiny")
logger.setLevel(logging.INFO)


class ESPToolchain:
  """
  esp toolchain - idf docker
  """

  def __init__(self, port, docker_image_name="espressif/idf:release-v5.4"):

    # arguments
    self.port = port
    self.docker_image_name = docker_image_name

    # docker
    self.dc = docker.from_env()


  def setup(self):
    """
    setup api
    """

    # reference images
    images = self.dc.images.list(filters={"reference": self.docker_image_name})

    # we already pulled it
    if images: return

    # TODO: this progress tracking is near useless
    #   but multiple tqdm bars make pycharm console go crazy
    for line in tqdm(self.dc.api.pull(f"docker.io/{self.docker_image_name}", stream=True, decode=True), desc="Pulling image", leave=True, file=sys.stdout,):
      pass


  def _create_container(self, command, volumes: Optional[list]=None, **kwargs):
    """
    create container
    """
    return self.dc.containers.run(
      image=self.docker_image_name,
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


  def set_target(self, src_path: Path, target='esp32-s3'):
    """
    compile for esp
    """

    # assertions
    assert target in ['esp32-s3'], "Sorry we do not support your selected target: [{}]".format(target)

    # create container
    container = self._create_container(
      "idf.py set-target {}".format(target),
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


  def compile(self, src_path: Path):
    """
    compile for esp
    """

    # resolve path
    src_path = src_path.resolve()

    # create container
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
    """
    flash esp
    """

    # resolve path
    src_path = src_path.resolve()

    # create container 
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
    """
    monitor code on esp
    """

    # resolve path
    src_path = src_path.resolve()
    
    # create container
    container = self._create_container(
      command=f"idf.py -p {self.port} monitor",
      volumes=[f"{str(src_path)}:/project"],
      devices=[f"{self.port}:{self.port}"],
      tty=True,
    )

    collected_lines = []
    
    try:
      output = container.attach(stdout=True, stream=True, logs=True)
      for line in self.line_accumulator(output):
        decoded = line.decode("utf-8").rstrip()
        print(decoded)
        collected_lines.append(decoded)
        if "main_task: Returned from app_main()" in decoded:
          parse_monitor_output(collected_lines)
          container.stop()
    except (Exception, KeyboardInterrupt):
      if collected_lines:
        parse_monitor_output(collected_lines)
      container.stop()




  def line_accumulator(self, byte_iterable):
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