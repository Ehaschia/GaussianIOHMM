from multiprocessing import Pool
import paramiko
import time
from collections import deque
import os
THREAD = 18
script_path = ''
cli_prefix = ''

def runner(idx, cli):
    client = paramiko.SSHClient()
    client.get_host_keys()
    client.connect('node' + idx)
    stdin, stdout, stderr = client.exec_command(cli_prefix + cli)
    return idx, cli


def multiprocess(configs, thread):
    thread = min(len(configs), thread)
    pool = Pool(processes=thread)
    runnings = []
    finished = []
    for i in range(thread):
        runnings.append(pool.apply_async(runner, (i, configs.pop())))
    while len(finished) != len(configs):
        time.sleep(30)
        for res in runnings:
            if res.ready():
                idx, cli = res.get()
                finished.append(cli)
                print(cli + " Finished")
                runnings.remove(res)
                if len(configs) > 0:
                    runnings.append(pool.apply_async(runner, (idx, configs.pop())))
            else:
                pass


if __name__ == '__main__':
    # freeze_support()
    configs = deque(os.listdir(script_path))
    multiprocess(configs, THREAD)
