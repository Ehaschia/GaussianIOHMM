import os, sys

import matplotlib

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

file_name = '2020-01-18_021855'


def read_log(root_path, file_name):
    logs = []
    with open(root_path + '/output/' + file_name + '/info.log', 'r') as f:
        for line in f.readlines():
            logs.append(line.strip())
    return logs


def get_info(log):
    infos = []
    for index in range(len(log)):
        if log[index].find("Epoch") != -1 and log[index].find('Best Epoch:') == -1:
            info = log[index].split(' ')
            epoch = info[8]
            loss = info[10]
            info = log[index+1].split(' ')
            dev = info[-1]
            infos.append((epoch, loss, dev))
    for info in infos:
        print('\t'.join(info))

    return infos


if __name__ == '__main__':
    log = read_log(root_path, file_name)
    get_info(log)