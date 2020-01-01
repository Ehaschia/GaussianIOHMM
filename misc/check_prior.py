import torch
from torch.optim import Adam
import torch.nn as nn
from torch.nn.parameter import Parameter
from model.basic_operation import atma
from model.inverse_wishart_distribution import InvWishart
from scipy.stats import invwishart
import numpy as np


class CheckPrior(nn.Module):
    def __init__(self, dim):
        super(CheckPrior, self).__init__()
        self.dim = dim
        var = invwishart.rvs(self.dim, np.eye(self.dim) / self.dim, size=1, random_state=None)
        self.var = Parameter(torch.from_numpy(np.array(var)).float(), requires_grad=True)

    def forward(self, beta):
        cho = torch.cholesky(self.var)
        main_loss = torch.trace(torch.abs(cho))
        log = InvWishart.logpdf(cho.unsqueeze(0), self.dim, torch.eye(self.dim) / self.dim)
        return main_loss - beta * log


configs = [{'epoch': 100, 'dim': 2, 'weight': 0.0},
           {'epoch': 100, 'dim': 2, 'weight': 0.01},
           {'epoch': 100, 'dim': 2, 'weight': 0.05},
           {'epoch': 100, 'dim': 2, 'weight': 0.1},
           {'epoch': 100, 'dim': 2, 'weight': 0.2},
           {'epoch': 100, 'dim': 2, 'weight': 0.5},
           {'epoch': 100, 'dim': 2, 'weight': 1.0},
           {'epoch': 100, 'dim': 2, 'weight': 10.0},
           {'epoch': 100, 'dim': 5, 'weight': 0.0},
           {'epoch': 100, 'dim': 5, 'weight': 0.01},
           {'epoch': 100, 'dim': 5, 'weight': 0.05},
           {'epoch': 100, 'dim': 5, 'weight': 0.1},
           {'epoch': 100, 'dim': 5, 'weight': 0.2},
           {'epoch': 100, 'dim': 5, 'weight': 0.5},
           {'epoch': 100, 'dim': 5, 'weight': 1.0},
           {'epoch': 100, 'dim': 5, 'weight': 10.0},
           {'epoch': 100, 'dim': 10, 'weight': 0.0},
           {'epoch': 100, 'dim': 10, 'weight': 0.01},
           {'epoch': 100, 'dim': 10, 'weight': 0.05},
           {'epoch': 100, 'dim': 10, 'weight': 0.1},
           {'epoch': 100, 'dim': 10, 'weight': 0.2},
           {'epoch': 100, 'dim': 10, 'weight': 0.5},
           {'epoch': 100, 'dim': 10, 'weight': 1.0},
           {'epoch': 100, 'dim': 10, 'weight': 10.0},
           {'epoch': 100, 'dim': 15, 'weight': 0.0},
           {'epoch': 100, 'dim': 15, 'weight': 0.01},
           {'epoch': 100, 'dim': 15, 'weight': 0.05},
           {'epoch': 100, 'dim': 15, 'weight': 0.1},
           {'epoch': 100, 'dim': 15, 'weight': 0.2},
           {'epoch': 100, 'dim': 15, 'weight': 0.5},
           {'epoch': 100, 'dim': 15, 'weight': 1.0},
           {'epoch': 100, 'dim': 15, 'weight': 10.0},
           {'epoch': 1000, 'dim': 2, 'weight': 0.0},
           {'epoch': 1000, 'dim': 2, 'weight': 0.01},
           {'epoch': 1000, 'dim': 2, 'weight': 0.05},
           {'epoch': 1000, 'dim': 2, 'weight': 0.1},
           {'epoch': 1000, 'dim': 2, 'weight': 0.2},
           {'epoch': 1000, 'dim': 2, 'weight': 0.5},
           {'epoch': 1000, 'dim': 2, 'weight': 1.0},
           {'epoch': 1000, 'dim': 2, 'weight': 10.0},
           {'epoch': 1000, 'dim': 5, 'weight': 0.0},
           {'epoch': 1000, 'dim': 5, 'weight': 0.01},
           {'epoch': 1000, 'dim': 5, 'weight': 0.05},
           {'epoch': 1000, 'dim': 5, 'weight': 0.1},
           {'epoch': 1000, 'dim': 5, 'weight': 0.2},
           {'epoch': 1000, 'dim': 5, 'weight': 0.5},
           {'epoch': 1000, 'dim': 5, 'weight': 1.0},
           {'epoch': 1000, 'dim': 5, 'weight': 10.0},
           {'epoch': 1000, 'dim': 10, 'weight': 0.0},
           {'epoch': 1000, 'dim': 10, 'weight': 0.01},
           {'epoch': 1000, 'dim': 10, 'weight': 0.05},
           {'epoch': 1000, 'dim': 10, 'weight': 0.1},
           {'epoch': 1000, 'dim': 10, 'weight': 0.2},
           {'epoch': 1000, 'dim': 10, 'weight': 0.5},
           {'epoch': 1000, 'dim': 10, 'weight': 1.0},
           {'epoch': 1000, 'dim': 10, 'weight': 10.0},
           {'epoch': 1000, 'dim': 15, 'weight': 0.0},
           {'epoch': 1000, 'dim': 15, 'weight': 0.01},
           {'epoch': 1000, 'dim': 15, 'weight': 0.05},
           {'epoch': 1000, 'dim': 15, 'weight': 0.1},
           {'epoch': 1000, 'dim': 15, 'weight': 0.2},
           {'epoch': 1000, 'dim': 15, 'weight': 0.5},
           {'epoch': 1000, 'dim': 15, 'weight': 1.0},
           {'epoch': 1000, 'dim': 15, 'weight': 10.0},
           ]

for config in configs:
    cnt = 0
    for i in range(1000):
        try:
            model = CheckPrior(config['dim'])
            optimizer = Adam(model.parameters(), lr=0.001)
            optimizer.zero_grad()

            for i in range(config['epoch']):
                optimizer.zero_grad()
                loss = model.forward(config['weight'])
                loss.backward()
                optimizer.step()
                # if i == 0 or i == 999:
                #     print(loss.item())
                #     print(model.var.detach().numpy())
        except:
            cnt += 1
    print("-"*20)
    print(config)
    print(1000 - cnt)
# dim = 2
# model = CheckPrior(dim)
# optimizer = Adam(model.parameters(), lr=0.01)
# optimizer.zero_grad()
#
# import copy
# before = copy.copy(model.var.detach().numpy())
# blogpdf = invwishart.logpdf(before, dim, np.eye(dim) / dim)
# for i in range(1000):
#     optimizer.zero_grad()
#     loss = model.forward(1.0)
#     loss.backward()
#     optimizer.step()
#     if i == 0 or i == 99:
#         print(loss.item())
# after = copy.copy(model.var.detach().numpy())
# alogpdf = invwishart.logpdf(after, dim, np.eye(dim) / dim)
# model.eval()
# with torch.no_grad():
#     loss = model.forward(1.0)
#     print(loss.item())
# print("-"*10)
# print(before)
# print(blogpdf)
# print("-"*10)
# print(after)
# print(alogpdf)

