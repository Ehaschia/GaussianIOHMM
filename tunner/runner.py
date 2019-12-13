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
        print('Task ' + cli + ' Done')
    else:
        print('Error ' + str(exit_status) + 'of ' + cli)
    client.close()
    return idx, cli


def multiprocess(configs, thread):
    thread = min(len(configs), thread)
    pool = Pool(processes=thread)
    runnings = []
    finished = []
    for i in range(1, thread + 1):
        time.sleep(0.1)
        runnings.append(pool.apply_async(runner, (i, configs.pop())))
    while len(finished) != len(configs):
        time.sleep(10)
        for res in runnings:
            if res.ready():
                idx, cli = res.get()
                finished.append(cli)
                print(str(idx) + ' ' + cli + ' Finished')
                runnings.remove(res)
                if len(configs) > 0:
                    runnings.append(pool.apply_async(runner, (idx, configs.pop())))
            else:
                pass


if __name__ == '__main__':
    # freeze_support()
    configs = deque(os.listdir(script_path))
    print(len(configs))
    multiprocess(configs, THREAD)
