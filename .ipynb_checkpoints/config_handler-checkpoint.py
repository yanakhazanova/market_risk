import yaml
import os

def load_config(config_path):
  with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
  return config


def create_folders(path):
  if not os.path.exists(path):
    os.makedirs(path)


def handle_configs(config):
  data_config = load_config(config)
  if 'folders' in data_config:
    for key in data_config:
      if isinstance(key, str):
        create_folders(data_config[key])
      else:
        parent, child = list(key.items())[0]
        create_folders(data_config[parent][child])

  return data_config