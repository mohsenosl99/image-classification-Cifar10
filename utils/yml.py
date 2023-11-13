import yaml
import os
import sys 
def load_config(config_name='config.yml'):
    path=config_name
    with open(os.path.join(sys.path[0], path)) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


