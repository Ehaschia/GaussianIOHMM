import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
from torch.autograd.function import Function


class SigmaInverse(Function):
    @staticmethod
    def forward(ctx, var0, var1):
        lambda0 = torch.inverse(var0)
        lambda1 = torch.inverse(var1)
        lambda_new = lambda0 + lambda1
        v_new = torch.inverse(lambda_new)
        ctx.save_for_backward(var0, var1, lambda0, lambda1, lambda_new, v_new)
        return v_new

    @staticmethod
    def backward(ctx, grad_output):
        var0, var1, lambda0, lambda1, lambda_new, v_new = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_v0_part = torch.matmul(v_new, lambda0)
        grad_v0 = torch.matmul(grad_v0_part.transpose(0, 1),
                               torch.matmul(grad_input, grad_v0_part))
        grad_v1_part = torch.matmul(v_new, lambda0)
        grad_v1 = torch.matmul(grad_v1_part.transpose(0, 1),
                               torch.matmul(grad_input, grad_v1_part))
        return grad_v0, grad_v1


class sigmaTestA(nn.Module):
    def __init__(self, np_v1, np_v2):
        super(sigmaTestA, self).__init__()
        self.v1 = Parameter(torch.from_numpy(np_v1))
        self.v2 = Parameter(torch.from_numpy(np_v2))
        self.sigma_inverse = SigmaInverse.apply

    def forward(self):
        v = self.sigma_inverse(self.v1, self.v2)
        # calculate the loss

        loss = torch.sum(v)
        return loss


class sigmaTestB(nn.Module):
    def __init__(self, np_v1, np_v2):
        super(sigmaTestB, self).__init__()
        self.v1 = Parameter(torch.from_numpy(np_v1))
        self.v2 = Parameter(torch.from_numpy(np_v2))

    def forward(self):
        lam1 = torch.inverse(self.v1)
        lam2 = torch.inverse(self.v2)
        lam = lam1 + lam2
        v = torch.inverse(lam)
        # calculate the loss
        loss = torch.sum(v)
        return loss


class Mu2SigmaTestA(nn.Module):
    def __init__(self, np_mu1, np_mu2, np_v1, np_v2):
        super(Mu2SigmaTestA, self).__init__()
        self.mu1 = Parameter(torch.from_numpy(np_mu1))
        self.mu2 = Parameter(torch.from_numpy(np_mu2))
        self.v1 = Parameter(torch.from_numpy(np_v1))
        self.v2 = Parameter(torch.from_numpy(np_v2))
        self.grads = {}

    def get_grads(self, name):
        def hook(grad):
            self.grads[name] = grad.numpy()

        return hook

    def forward(self):
        lam1 = torch.inverse(self.v1)
        lam2 = torch.inverse(self.v2)
        lam = lam1 + lam2
        v = torch.inverse(lam)
        v_lam1 = torch.matmul(v, lam1)
        mu = torch.matmul(v_lam1, self.mu1)
        mu.register_hook(self.get_grads("mu"))
        v_lam1.register_hook(self.get_grads("v_lam1"))
        v.register_hook(self.get_grads("v"))
        lam.register_hook(self.get_grads("lam"))
        lam1.register_hook(self.get_grads("lam1"))
        lam2.register_hook(self.get_grads("lam2"))
        loss = torch.matmul(self.mu2.transpose(0, 1), mu).squeeze()
        return loss


class InverseDet(nn.Module):
    def __init__(self, v1_np, v2_np, x1_np, x2_np):
        super(InverseDet, self).__init__()
        self.v1 = Parameter(torch.from_numpy(v1_np))
        self.v2 = Parameter(torch.from_numpy(v2_np))
        self.x1 = Parameter(torch.from_numpy(x1_np))
        self.x2 = Parameter(torch.from_numpy(x2_np))
        self.grads = {}

    def get_grads(self, name):
        def hook(grad):
            self.grads[name] = grad

        return hook

    def reture_grads(self):
        return self.grads

    def forward(self):
        lam1 = torch.inverse(self.v1)
        lam2 = torch.inverse(self.v2)

        lam_new = lam1 + lam2

        v_new = torch.inverse(lam_new)
        v_new.register_hook(self.get_grads("v_new"))
        lam_new.register_hook(self.get_grads("lam_new"))
        lam1.register_hook(self.get_grads("lam1"))
        lam2.register_hook(self.get_grads("lam2"))
        loss = torch.matmul(self.x1, torch.matmul(v_new, self.x2)).squeeze()
        return loss


def generate_v(n, scale):
    v_np = np.random.rand(n, n)
    v_np = np.triu(v_np) + scale * np.eye(n)
    v_np = np.matmul(v_np.transpose(), v_np)
    return v_np


# def det_grad(v1, v2):
#     np_lam1 = np.linalg.inv(v1)
#     np_lam2 = np.linalg.inv(v2)
#     np_lam = np_lam1 + np_lam2
#     np_v = np.linalg.inv(np_lam)
#     det_v = np.linalg.det(np_v)
#     return -1.0 * det_v * np_v

def bilinear_grad(x1, x2):
    x1t = x1.transpose()
    x2t = x2.transpose()
    return np.matmul(x1t, x2t)


# # for det
# def v1_grad(v1, v2):
#     np_lam1 = np.linalg.inv(v1)
#     np_lam2 = np.linalg.inv(v2)
#     np_lam = np_lam1 + np_lam2
#     np_v = np.linalg.inv(np_lam)
#     det_v = np.linalg.det(np_v)
#     grad = np.matmul(np_lam1, np.matmul(-1.0 * det_v * np_v, np_lam1))
#     # grad = np.matmul(np_lam1, np.matmul(np_lam1, -1.0 * det_v * np_v))
#     return grad


def v1_grad(v1, v2, x1, x2):
    np_lam1 = np.linalg.inv(v1)
    np_lam2 = np.linalg.inv(v2)
    np_lam = np_lam1 + np_lam2
    np_v = np.linalg.inv(np_lam)
    # np_loss = np.matmul(np.matmul(x1, np_v), x2).squeeze_()
    bg = bilinear_grad(x1, x2)
    part1 = np.matmul(np_v, np_lam1)
    grad = np.matmul(part1.transpose(), np.matmul(bg, part1))
    return grad


def testSigma():
    np.random.seed(2)
    torch.manual_seed(1)
    n = 2
    scale = 1e-2
    v1_np = generate_v(n, scale)
    v2_np = generate_v(n, scale)
    x1_np = np.random.rand(1, n)
    x2_np = np.random.rand(n, 1)

    modelA = sigmaTestA(v1_np, v2_np)
    modelB = sigmaTestB(v1_np, v2_np)

    optimzerA = optim.SGD(modelA.parameters(), lr=0.1, nesterov=False, momentum=0.0)
    optimzerB = optim.SGD(modelB.parameters(), lr=0.1, nesterov=False, momentum=0.0)

    optimzerA.zero_grad()
    modelA.train()

    lossA = modelA.forward()

    lossA.backward()
    v1_gradA = modelA.v1.grad.numpy()

    optimzerB.zero_grad()
    modelB.train()
    lossB = modelB.forward()
    lossB.backward()
    v1_gradB = modelB.v1.grad.numpy()

    print(v1_gradB - v1_gradA)


def mu2sigma1_grad(mu1, mu2, v1, v2, ground_truth=None):
    # forward part
    lam1 = np.linalg.inv(v1)
    lam2 = np.linalg.inv(v2)
    lam = lam1 + lam2
    v = np.linalg.inv(lam)
    mu = np.matmul(v, np.matmul(lam1, mu1))
    loss = np.matmul(mu2.transpose(), mu).squeeze()

    # part1
    mu_grad = mu2
    v_lam1_grad = np.matmul(mu_grad, mu1.transpose())

    part1 = np.matmul(v, lam1)

    v1_v_grad = np.matmul(np.matmul(part1.transpose(),
                                    np.matmul(v_lam1_grad, lam1)), part1)
    v1_lam1_grad = np.matmul(lam1.transpose(),
                             np.matmul(np.matmul(v.transpose(), v_lam1_grad), lam1))

    v1_grad = v1_v_grad - v1_lam1_grad
    dim = mu1.size
    v1_grad_v2 = part1 - np.eye(dim)
    v1_grad_v2 = np.matmul(part1.transpose(), np.matmul(np.matmul(v_lam1_grad, lam1), v1_grad_v2))

    # v2 gradient
    v_grad = np.matmul(v_lam1_grad, lam1.transpose())
    part2 = np.matmul(v, lam2)
    v2_grad = np.matmul(part2.transpose(), np.matmul(v_grad, part2))
    return v1_grad


def testMu2Sigma():
    np.random.seed(0)
    torch.manual_seed(0)
    n = 2
    scale = 1e-2
    v1_np = generate_v(n, scale)
    v2_np = generate_v(n, scale)
    mu1_np = np.random.rand(n, 1)
    mu2_np = np.random.rand(n, 1)
    model = Mu2SigmaTestA(mu1_np, mu2_np, v1_np, v2_np)
    optimizer = optim.SGD(model.parameters(), lr=0.1, nesterov=False, momentum=0.0)
    optimizer.zero_grad()
    model.train()

    loss = model.forward()

    loss.backward()

    grads = model.grads
    grads['v1'] = model.v1.grad.numpy()
    grads['v2'] = model.v2.grad.numpy()
    grads['mu1'] = model.mu1.grad.numpy()
    grads['mu2'] = model.mu2.grad.numpy()

    v1_grad = mu2sigma1_grad(mu1_np, mu2_np, v1_np, v2_np, grads)
    print(v1_grad)


if __name__ == '__main__':
    # testSigma()
    testMu2Sigma()
