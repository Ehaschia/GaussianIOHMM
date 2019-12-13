from multiprocessing import Pool
import paramiko
import time
from collections import deque
import os

THREAD = 18
script_path = '/public/home/tukewei/hanwj/zlw/GaussianIOHMM/scripts'
cli_prefix = 'sh /public/home/tukewei/hanwj/zlw/GaussianIOHMM/scripts/'


def runner(idx, cli):
    client = paramiko.SSHClient()
    client.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
    client.connect('node' + str(idx))
    print(cli_prefix + cli)
    stdin, stdout, stderr = client.exec_command(cli_prefix + cli)
    exit_status = stdout.channel.recv_exit_status()  # Blocking call
    if exit_status == 0:
        print('Task ' + cli + ' Done at Thread ' + str(idx))
    else:
        print('Error ' + str(exit_status) + 'of ' + cli + ' at Thread ' + str(idx))
    client.close()
    return idx, cli


def multiprocess(configs, thread):
    thread = min(len(configs), thread)
    pool = Pool(processes=thread)
    runnings = {}
    finished = []
    # init threads
    for i in range(1, thread + 1):
        time.sleep(0.1)
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
        print("Activate thread: " + str(activate_thread))


if __name__ == '__main__':
    # freeze_support()
    configs = deque(os.listdir(script_path))
    print(len(configs))
    multiprocess(configs, THREAD)
