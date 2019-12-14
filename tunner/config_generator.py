import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)
from copy import copy
from tunner.utils import ROOT_DIR, param_json2list, load_done_configs

root = ROOT_DIR + '/scripts'
if not os.path.exists(root):
    os.makedirs(root)

prefix = '#!/usr/bin/env bash\n' \
         'cd ' + ROOT_DIR + '\n'
python = '/public/home/tukewei/hanwj/edus/anaconda2/envs/iohmm/bin/python'
file = 'sequence_labeling.py'
suffix = ''

# parameters
parameters = {
    'data': ['./dataset/syntic_data_yong/0-1000-10-new', './dataset/syntic_data_yong/0-3000-10-new',
             './dataset/syntic_data_yong/0-10000-10-new', './dataset/syntic_data_yong/0-30000-10-new',
             './dataset/syntic_data_yong/0-10000-10-new'],
    'batch': ['8', '32', '128'],
    'dim': ['10'],
    'random_seed': ['1', '2', '3'],
    'in_mu_drop': ['0.0', '0.2', '0.5'],
    't_mu_drop': ['0.0', '0.2', '0.5'],
    'out_mu_drop': ['0.0', '0.2', '0.5'],
    'trans_cho_method': ['random', 'wishart'],
    'input_cho_init': ['0.0', '1.0'],
    'output_cho_init': ['0.0', '1.0'],
}

keys = list(parameters.keys())
configs = []


def dfs(config, idx):
    for value in parameters[keys[idx]]:
        new_config = copy(config)
        new_config.append('--' + keys[idx] + ' ' + value)
        if idx + 1 == len(keys):
            configs.append(new_config)
        else:
            dfs(new_config, idx + 1)


def config_generate(config):
    return prefix + '\n' + python + ' ' + file + ' ' + ' '.join(config) + '\n' + suffix


def configs_generate():
    for idx in range(len(configs)):
        with open(root + '/' + str(idx).zfill(len(str(len(configs)))) + '.sh', 'w') as f:
            f.write(config_generate(configs[idx]))


def done_filter(root_path, generate_configs):
    done_configs = load_done_configs(root_path)
    for done_config in done_configs:
        config_list = param_json2list(root_path + done_config, keys)
        generate_configs.remove(config_list)
    return generate_configs


if __name__ == '__main__':

    dfs([], 0)
    generate_configs = done_filter(ROOT_DIR + '/output/', configs)
    configs_generate()
