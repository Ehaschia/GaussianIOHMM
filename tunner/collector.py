import json
import os
from tunner.utils import ROOT_DIR
import csv

output_dir = ROOT_DIR + '/output/'
finished_list = list(filter(lambda x: os.path.exists(output_dir + x + '/result.json'), os.listdir(output_dir)))
outer = open(output_dir + 'all.csv', 'w')
writer = csv.writer(outer)
writer.writerow(['data', 'batch', 'lr', 'var_scale', 'dim', 'random_seed', 'in_mu_drop', 'in_cho_drop', 't_mu_drop',
                 't_cho_drop', 'out_mu_drop', 'out_cho_drop', 'trans_cho_method', 'input_cho_init', 'trans_cho_init',
                 'output_cho_init', 'input_comp_num', 'tran_comp_num', 'output_comp_num', 'max_comp', 'Epoch', 'Dev',
                 'Test'])


for finished_name in finished_list:
    with open(output_dir + finished_name + '/param.json') as f:
        params = json.load(f)
    with open(output_dir + finished_name + '/result.json') as f:
        result = json.load(f)

    writer.writerow([params['data'], params['batch'], params['lr'], params['var_scale'], params['dim'],
                     params['random_seed'], params['in_mu_drop'], params['in_cho_drop'], params['t_mu_drop'],
                     params['t_cho_drop'], params['out_mu_drop'], params['out_cho_drop'], params['trans_cho_method'],
                     params['input_cho_init'], params['trans_cho_init'], params['output_cho_init'],
                     params['input_comp_num'], params['tran_comp_num'], params['output_comp_num'],
                     params['max_comp'], result['Epoch'], result['Dev'], result['Test']])
outer.close()
