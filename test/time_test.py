import numpy as np

import time

np.random.seed(0)

dim = 50
batch = 10000

mtime = 0.0
itime = 0.0

for i in range(batch):
    D = np.random.rand(dim)
    D = np.diag(D)
    A = np.random.rand(dim, dim)
    A = np.tril(A) + 0.5 * np.eye(50)

    mtimes = time.time()
    M = np.matmul(D, A)
    mtime += (time.time() - mtimes)

    itimes = time.time()
    I_inv = np.linalg.inv(A + D)
    itime += (time.time() - itimes)

print(mtime)
print(itime)