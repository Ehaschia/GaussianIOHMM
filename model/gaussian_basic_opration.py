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

    aligned_eta = eta.unsqueeze(-1)
    if mu is not None:
        aligned_mu = mu.unsqueeze(-1)
    else:
        aligned_mu = None

    dim = lam.size(-1)
    part1 = dim * np.log(np.pi * 2)

    part2 = torch.logdet(lam)

    if mu is not None:
        if len(aligned_eta.size()) > 1:
            part3 = torch.matmul(aligned_eta.transpose(-2, -1), aligned_mu).reshape(aligned_eta.size()[:-2])
        else:
            part3 = torch.matmul(aligned_eta, torch.matmul(sig, aligned_eta))
    else:
        if sig is None:  # and mu is None:
            sig = torch.inverse(lam)
        part3 = torch.matmul(aligned_eta.transpose(-2, -1), torch.matmul(sig, aligned_eta)).reshape_as(part2)

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

    lambda_new = lambda0 + lambda1_pad
    eta_new = eta0 + eta1_pad

    sigma_new = torch.inverse(lambda_new)
    mu_new = torch.matmul(sigma_new, eta_new.unsqueeze(-1)).squeeze(-1)

    if need_zeta:
        zeta0 = calculate_zeta(eta0, lambda0, mu=mu0)
        zeta1 = calculate_zeta(eta1, lambda1, mu=mu1)
        zeta_new = calculate_zeta(eta_new, lambda_new, sig=sigma_new)
        scale = zeta0 + zeta1 - zeta_new
    else:
        scale = None

    if forward:
        res_mu = mu_new[:, :, :, dim1:]
        res_sigma = sigma_new[:, :, :, dim1:, dim1:]
    else:
        res_mu = mu_new[:, :, :, dim1]
        res_sigma = sigma_new[:, :, :, dim1, :dim1]

    return scale, res_mu, res_sigma


def gaussian_top_k_pruning(score, mu, var, k=1):
    if score.size()[-1] <= k:
        return score, mu, var
    pruned_score, index = torch.topk(score, k, dim=-1)
    batch, _, dim = mu.size()
    pruned_mu = torch.gather(mu, 1, index.view(batch, 1, 1).expand(batch, 1, dim))
    pruned_var = torch.gather(var, 1, index.view(batch, 1, 1, 1).repeat(1, 1, dim, dim))
    return pruned_score, pruned_mu, pruned_var
