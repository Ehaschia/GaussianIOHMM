import torch
from torch.autograd.function import Function
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.nn import Parameter

np.random.seed(0)
torch.manual_seed(0)


class GaussianMultiply(Function):
    # def __init__(self):
    #     super(GaussianMultiply, self).__init__()

    @staticmethod
    def forward(ctx, mu0, mu1, cho0, cho1):
        """
        calculate of two gaussian. Here gaussian represent as (mu, chokesky),
        here var be represent as cholesky^T cholesky, heew cholesky is upper triangular matrix
        :param ctx: holder
        :param mu0: mu for gaussian 1
        :param mu1: mu for gaussian 2
        :param cho0: cholesky matrix for gaussian 1
        :param cho1: cholesky matrix for gaussian 2
        :return: scale: the scale of two gaussian multiply
        :return: mu_new: the mu for the new gaussian
        :return: cho_new: the chokesky matrix of the new gaussian
        """

        dim = mu0.size(-1)
        # inverse
        # TODO here inverse calculation can use numpy or pytorch, here we conside torch first.
        # lambda0 = np.linalg.inv(var0)
        # lambda1 = np.linalg.inv(var1)
        eye = torch.eye(dim).unsqueeze_(0).double()
        lambda0 = torch.cholesky_solve(cho0, eye, upper=True)
        lambda1 = torch.cholesky_solve(cho1, eye, upper=True)

        mu0 = mu0.unsqueeze(-1)
        mu1 = mu1.unsqueeze(-1)
        eta0 = torch.matmul(lambda0, mu0)
        eta1 = torch.matmul(lambda1, mu1)

        # calculate zeta
        diag0 = torch.diagonal(cho0, dim1=1, dim2=2)
        diag1 = torch.diagonal(cho1, dim1=1, dim2=2)
        zeta0 = -0.5 * (dim * np.log(np.pi * 2) -
                        torch.sum(torch.log(diag0 * diag0), dim=-1)
                        + mu0.transpose(1, 2).matmul(eta0).reshape(-1))

        zeta1 = -0.5 * (dim * np.log(np.pi * 2) -
                        torch.sum(torch.log(diag1 * diag1), dim=-1)
                        + mu1.transpose(1, 2).matmul(eta1).reshape(-1))

        lambda_new = lambda0 + lambda1
        eta_new = eta0 + eta1

        var_new = torch.inverse(lambda_new)
        cho_new = torch.cholesky(var_new, upper=True)

        mu_new = torch.matmul(var_new, eta_new)
        diag_new = torch.diagonal(cho_new, dim1=1, dim2=2)
        zeta_new = -0.5 * (dim * np.log(np.pi * 2) -
                           torch.sum(torch.log(diag_new * diag_new), dim=-1)
                           + mu_new.transpose(1, 2).matmul(eta_new).reshape(-1))

        scale = zeta0 + zeta1 - zeta_new


        return scale, mu_new, cho_new

    @staticmethod
    def backward(ctx, *grad_outputs):
        scale_gradient, mu_gradient, cho_gradient = grad_outputs
        

class GaussianMultiScore(nn.Module):
    def __init__(self, mu0, mu1, cho0, cho1, sample):
        super(GaussianMultiScore, self).__init__()
        self.gaussian_multiply = GaussianMultiply.apply
        self.mu0 = Parameter(mu0)
        self.mu1 = Parameter(mu1)
        self.cho1 = Parameter(cho1)
        self.cho0 = Parameter(cho0)
        self.sample = Parameter(sample)

    def gaussian_score(self, mu, cho, sample):
        dim = mu.size(-1)
        var = torch.matmul(cho.transpose(1, 2), cho)

        var_inv = torch.inverse(var)

        exp_part = -0.5 * torch.bilinear((mu - sample), (mu - sample), var_inv, None).reshape(-1)
        det = torch.diagonal(cho, dim1=1, dim2=2)
        det = torch.sum(det * det, dim=1)
        scale_part = torch.sqrt(((2 * np.pi) ** dim) * det)
        scale = (1.0 / scale_part) * torch.exp(exp_part)
        return scale

    def forward(self):
        scale, mu, cho = self.gaussian_multiply(self.mu0, self.mu1, self.cho0, self.cho1)
        sample_score = self.gaussian_score(mu, cho, self.sample)
        return torch.exp(scale) * sample_score


mu0 = torch.from_numpy(np.array([0.0]))
mu1 = torch.from_numpy(np.array([1.0]))
cho0 = torch.from_numpy(np.array([[1.0]]))
cho1 = torch.from_numpy(np.array([[1.0]]))

mu0.unsqueeze_(0)
mu1.unsqueeze_(0)
cho0.unsqueeze_(0)
cho1.unsqueeze_(0)

sample = torch.from_numpy(np.array([0.0]))
sample.unsqueeze_(0)
model = GaussianMultiScore(mu0, mu1, cho0, cho1, sample)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0, nesterov=False)
optimizer.zero_grad()
loss = model()
loss.backward()
optimizer.step()
