import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter


class DetTestA(nn.Module):
    def __init__(self, mat):
        super(DetTestA, self).__init__()
        self.mat = Parameter(torch.from_numpy(mat))

    def forward(self):
        return torch.log(torch.det(self.mat))


class DetTestB(nn.Module):
    def __init__(self, mat):
        super(DetTestB, self).__init__()
        self.mat = Parameter(torch.from_numpy(mat))

    def forward(self):
        cho = torch.cholesky(self.mat)
        score = 2.0 * torch.sum(torch.log(torch.abs(torch.diagonal(cho))))
        return score


def generate_v(n, scale):
    v_np = np.random.rand(n, n)
    v_np = np.triu(v_np) + scale * np.eye(n)
    v_np = np.matmul(v_np.transpose(), v_np)
    return v_np


def det_test():
    np.random.seed(0)
    torch.manual_seed(0)

    n = 5
    scale = 1e-1
    lr = 0.1
    mat = generate_v(n, scale)
    modelA = DetTestA(mat)
    optimA = torch.optim.SGD(modelA.parameters(), lr=lr, nesterov=False)
    optimA.zero_grad()
    lossA = modelA.forward()
    lossA.backward()
    gradA = modelA.mat.grad.numpy()

    modelB = DetTestB(mat)
    optimB = torch.optim.SGD(modelB.parameters(), lr=lr, nesterov=False)
    optimB.zero_grad()
    lossB = modelB.forward()
    lossB.backward()
    gradB = modelB.mat.grad.numpy()

    print(lossB.item() - lossA.item())
    print(gradA - gradB)


if __name__ == '__main__':
    det_test()
