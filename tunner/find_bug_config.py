import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

from tunner.utils import *

from tunner.config_generator import prefix, python, file, suffix, keys

configs = load_configs(ROOT_DIR + '/scripts')

# clean configs to list
cleaned_configs = {}
for config_pair in configs:
    config_name, config = config_pair
    config.replace(prefix, '')
    config.replace(python, '')
    config.replace(file, '').strip()
    if config not in cleaned_configs:
        cleaned_configs[config.strip()] = config_name
    else:
        raise ValueError("Config Conflict!")

# load un-done file
output_config_paths = os.listdir(ROOT_DIR + '/output')
undone_config_paths = filter(lambda x: not os.path.exists(x + 'result.json'), output_config_paths)

for undone_config_path in undone_config_paths:
    config = param_json2list(undone_config_path, keys)
    if ' '.join(config) in cleaned_configs:
        print('='*20)
        print('Undone:\t' + undone_config_path)
        print('Script:\t' + cleaned_configs[config])