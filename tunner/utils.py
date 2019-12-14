import json
import os

# from tunner.config_generator import json2list

# Hyper setting
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))


# load json
def load_done_configs(name):
    output_dir = os.path.join(ROOT_DIR, name)
    return filter(lambda x: os.path.exists(x + '/result.json'), os.listdir(output_dir))


def param_json2list(json_file_name, keys):
    with open(json_file_name, 'b') as f:
        config = json.load(f)
    res = []
    for key in keys:
        res.append('--' + key + ' ' + str(config[key]))
    return res


# load the generate config
def load_configs(path):
    configs = []
    config_paths = os.listdir(path)
    for config_path in config_paths:
        with open(config_path, 'r') as f:
            configs.append((config_path, f.read()))
    return configs
