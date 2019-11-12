from typing import Tuple

import torch
import numpy as np
import torch.nn.functional as F


def atma(cholesky: torch.Tensor) -> torch.Tensor:
    return torch.matmul(cholesky.transpose(-2, -1), cholesky)


def calculate_zeta(eta: torch.Tensor, lam: torch.Tensor,
                   mu: torch.Tensor = None, sig: torch.Tensor = None) -> torch.Tensor:
    # zeta in log format
    # zeta = - 0.5 * (dim * log 2pi  - log|lam| + eta^T * lam^{-1} * eta)
    # zeta = - 0.5 * (dim * log 2pi  + log|sigma| + eta^T * sigma * eta)
    # TODO optimize calculate process, such as determinate and inverse

    if eta.size()[-1] != 1:
        aligned_eta = eta.unsqueeze(-1)
    else:
        aligned_eta = eta
    if mu is not None and mu.size()[-1] != 1:
        aligned_mu = mu.unsqueeze(-1)
    else:
        aligned_mu = mu

    dim = lam.size(-1)
    part1 = dim * np.log(np.pi * 2)
    if len(lam.size()) == 2:
        part2 = torch.log(torch.det(lam))
    else:
        # part2 = 2.0 * torch.sum(torch.log(torch.abs(torch.diagonal(torch.cholesky(lam),
        #                                                            dim1=-1, dim2=-2))), dim=-1)
        part2 = torch.det(lam)

    if aligned_mu is not None:
        if len(aligned_eta.size()) > 1:
            part3 = torch.matmul(aligned_eta.transpose(-2, -1), aligned_mu).reshape(aligned_eta.size()[:-2])
        else:
            part3 = torch.matmul(aligned_eta, torch.matmul(sig, aligned_eta))
    else:
        if sig is None:  # and mu is None:
            sig = torch.inverse(lam)
        part3 = torch.matmul(aligned_eta.transpose(-2, -1), torch.matmul(sig, aligned_eta)).squeeze()

    return -0.5 * (part1 - part2 + part3)


# multiply two same dimension gaussian
# Input format should be [batch, dim]
def gaussian_multi(mu0: torch.Tensor, mu1: torch.Tensor,
                   var0: torch.Tensor, var1: torch.Tensor,
                   need_zeta=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lambda0 = torch.inverse(var0)
    lambda1 = torch.inverse(var1)

    eta0 = torch.matmul(lambda0, mu0.unsqueeze(-1)).squeeze(-1)
    eta1 = torch.matmul(lambda1, mu1.unsqueeze(-1)).squeeze(-1)

    lambda_new = lambda0 + lambda1
    eta_new = eta0 + eta1
    sigma_new = torch.inverse(lambda_new)
    mu_new = torch.matmul(sigma_new, eta_new.unsqueeze(-1)).squeeze(-1)
    if need_zeta:
        zeta0 = calculate_zeta(eta0, lambda0, mu=mu0)
        zeta1 = calculate_zeta(eta1, lambda1, mu=mu1)
        zeta_new = calculate_zeta(eta_new, lambda_new, mu=mu_new)

        score = zeta0 + zeta1 - zeta_new
    else:
        score = None
    return score, mu_new, sigma_new


# if forward, integral the upper half. Else, lower half.
# the first Gaussian is regarded as the emission/inside score.
# the second Gaussian is regarded as the transition score.
def gaussian_multi_integral(mu0: torch.Tensor, mu1: torch.Tensor,
                            var0: torch.Tensor, var1: torch.Tensor,
                            forward: bool = True, need_zeta=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dim0 = mu0.size()[-1]
    dim1 = mu1.size()[-1]

    assert dim0 > dim1

    if forward:
        mu_pad = (0, dim1)
        var_pad = (0, dim1, 0, dim1)
    else:
        mu_pad = (dim1, 0)
        var_pad = (dim1, 0, dim1, 0)

    lambda0 = torch.inverse(var0)
    lambda1 = torch.inverse(var1)

    # use new variable avoid in-place operation
    mu0_new = mu0.unsqueeze(-1)
    mu1_new = mu1.unsqueeze(-1)
    eta0 = torch.matmul(lambda0, mu0_new).squeeze(-1)
    eta1 = torch.matmul(lambda1, mu1_new).squeeze(-1)
    # mu0_new = mu0_new.squeeze(-1)

    eta1_pad = F.pad(eta1, mu_pad, 'constant', 0)
    lambda1_pad = F.pad(lambda1, var_pad, 'constant', 0)

    lambda_new = lambda0.unsqueeze(0) + lambda1_pad
    eta_new = eta0.unsqueeze(0) + eta1_pad

    sigma_new = torch.inverse(lambda_new)
    mu_new = torch.matmul(sigma_new, eta_new.unsqueeze(-1)).squeeze(-1)

    if need_zeta:
        zeta0 = calculate_zeta(eta0.unsqueeze(-1), lambda0, mu=mu0_new)
        zeta1 = calculate_zeta(eta1.unsqueeze(-1), lambda1, mu=mu1_new)
        zeta_new = calculate_zeta(eta_new, lambda_new, sig=sigma_new)

        scale = zeta0 + zeta1 - zeta_new
    else:
        scale = None

    if forward:
        res_mu = mu_new[:, dim1:]
        res_sigma = sigma_new[:, dim1:, dim1:]
    else:
        res_mu = mu_new[:, :dim1]
        res_sigma = sigma_new[:, :dim1, :dim1]

    return scale, res_mu, res_sigma
