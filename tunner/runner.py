import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

from multiprocessing import Pool
import paramiko
import time
from collections import deque

THREAD = 17
script_path = '/public/home/tukewei/hanwj/zlw/GaussianIOHMM/scripts'
cli_prefix = 'sh /public/home/tukewei/hanwj/zlw/GaussianIOHMM/scripts/'
skip = {17}


def runner(idx, cli):
    client = paramiko.SSHClient()
    client.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
    print("Begin to connect " + str(idx))
    client.connect('node' + str(idx))
    print("Connected " + str(idx))
    stdin, stdout, stderr = client.exec_command(cli_prefix + cli)
    print("Run " + cli + " on " + str(idx))
    exit_status = stdout.channel.recv_exit_status()  # Blocking call
    print("Finish " + cli + " on " + str(idx))
    if exit_status == 0:
        print('Task ' + cli + ' Done at Thread ' + str(idx))
    else:
        print('Error ' + str(exit_status) + 'of ' + cli + ' at Thread ' + str(idx))
    client.close()
    print("Close " + str(idx))
    return idx, cli


def multiprocess(configs, thread):
    thread = min(len(configs), thread)
    pool = Pool(processes=thread)
    runnings = {}
    finished = []
    # init threads
    for i in range(1, thread + 1 + len(skip)):
        time.sleep(1)
        if i in skip:
            continue
        runnings[i] = pool.apply_async(runner, (i, configs.pop()))

    while len(finished) != len(configs):
        time.sleep(10)
        activate_thread = 0
        for idx in runnings.keys():
            if runnings[idx] is not None:
                activate_thread += 1
                if runnings[idx].ready():
                    finished_thread = runnings[idx]
                    runnings[idx] = None
                    idx, cli = finished_thread.get()
                    finished.append(cli)
                    if len(configs) > 0:
                        runnings[idx] = pool.apply_async(runner, (idx, configs.pop()))
            else:
                pass


if __name__ == '__main__':
    # freeze_support()
    configs = deque(os.listdir(script_path))
    print(len(configs))
    multiprocess(configs, THREAD)
