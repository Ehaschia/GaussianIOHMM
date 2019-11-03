import torch
import torch.nn as nn
from torch.nn import Parameter


class SingleUnitRule(nn.Module):
    def __init__(self, dim: int) -> None:
        super(SingleUnitRule, self).__init__()
        self.mu = Parameter(torch.empty(dim), requires_grad=True)
        self.cholesky = Parameter(torch.empty(dim, dim), requires_grad=True)
        self.reset_parameter()
        self.var = self.cholesky.t().matmul(self.cholesky)
        
    def reset_parameter(self):
        nn.init.xavier_normal_(self.mu)
        nn.init.xavier_normal_(self.cholesky)
        self.cholesky.weight = torch.tril(self.cholesky.weight)
        

class DoubleUnitRule(nn.Module):
    def __init__(self, dim: int) -> None:
        super(DoubleUnitRule, self).__init__()
        self.mu = Parameter(torch.empty(2 * dim), requires_grad=True)
        self.cholesky = Parameter(torch.empty(2 * dim, 2 * dim), requires_grad=True)
        self.reset_parameter()
        self.var = self.cholesky.t().matmul(self.cholesky)

    def reset_parameter(self):
        nn.init.xavier_normal_(self.mu)
        nn.init.xavier_normal_(self.cholesky)
        self.cholesky.weight = torch.tril(self.cholesky.weight)
