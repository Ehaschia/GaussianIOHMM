import torch
import torch.nn as nn
from torch.nn import Parameter


class SingleUnitRule(nn.Module):
    def __init__(self, dim: int) -> None:
        self.mu = Parameter(torch.tensor(dim))
        self.cholesky = Parameter(torch.tensor(dim, dim))
        self.reset_parameter()
        self.var = self.cholesky.t().matmul(self.cholesky)
        
    def reset_parameter(self):
        nn.init.xavier_normal_(self.mu)
        nn.init.xavier_normal_(self.cholesky)
        self.cholesky.weight = torch.tril(self.cholesky.weight)
        

class DoubleUnitRule(nn.Module):
    def __init__(self, dim: int) -> None:
        self.mu = Parameter(torch.tensor(2 * dim))
        self.cholesky = Parameter(torch.tensor(2 * dim, 2 * dim))
        self.reset_parameter()
        self.var = self.cholesky.t().matmul(self.cholesky)

    def reset_parameter(self):
        nn.init.xavier_normal_(self.mu)
        nn.init.xavier_normal_(self.cholesky)
        self.cholesky.weight = torch.tril(self.cholesky.weight)
