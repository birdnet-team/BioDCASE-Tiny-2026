# --
# biodcase 2026 - tiny ml (task 3)


if __name__ == '__main__':
  """
  tiny ml starts here
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # info
  print("Hello Tiny ML 2026, version: {}".format(cfg['version']))