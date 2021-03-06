from typing import Tuple

import torch
import numpy as np
import torch.nn.functional as F


def reset_embedding(init_embedding, embedding_layer, embedding_dim, trainable, var=False):
    if init_embedding is None:
        scale = np.sqrt(3.0 / embedding_dim)
        if var:
            embedding_layer.weight.data.uniform_(-scale, scale)
        else:
            embedding_layer.weight.data.uniform_(0.1, scale)
    else:
        embedding_layer.load_state_dict({'weight': init_embedding})
    embedding_layer.weight.requires_grad = trainable


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


# used for diagonal gaussian, thus lam is [batch, dim]
def fast_calculate_zeta(eta: torch.Tensor, lam: torch.Tensor,
                        mu: torch.Tensor = None, sig: torch.Tensor = None) -> torch.Tensor:
    # zeta in log format
    # zeta = - 0.5 * (dim * log 2pi  - log|lam| + eta^T * lam^{-1} * eta)
    # zeta = - 0.5 * (dim * log 2pi  + log|sigma| + eta^T * sigma * eta)
    # TODO optimize calculate process, such as determinate and inverse

    dim = lam.size(-1)
    part1 = dim * np.log(np.pi * 2)

    part2 = torch.sum(torch.log(lam), dim=-1)

    if mu is not None:
        part3 = torch.matmul(eta.unsqueeze(-2), mu.unsqueeze(-1)).reshape_as(part2)
    else:
        if sig is None:  # and mu is None:
            sig = 1.0 / lam
        part3 = torch.chain_matmul(eta.unsqueeze(-2), sig, eta.unsqueeze(-1)).reshape_as(part2)

    return -0.5 * (part1 - part2 + part3)


# multiply two same dimension gaussian
# Input format should be [..., dim] and [..., dim, dim] for mu and var, respectively.
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


# diag0 and diag1 is if using high speed inverse
def fast_gaussian_multi(mu0: torch.Tensor, mu1: torch.Tensor,
                        var0: torch.Tensor, var1: torch.Tensor,
                        need_zeta=True, diag0=False, diag1=False):
    if diag0:
        lambda0 = 1.0 / var0
        eta0 = lambda0 * mu0
    else:
        lambda0 = torch.inverse(var0)
        eta0 = torch.matmul(lambda0, mu0.unsqueeze(-1)).squeeze(-1)

    if diag1:
        lambda1 = 1.0 / var1
        eta1 = lambda1 * mu1
    else:
        lambda1 = torch.inverse(var1)
        eta1 = torch.matmul(lambda1, mu1.unsqueeze(-1)).squeeze(-1)

    # xor to get same shape
    if not diag0 ^ diag1:
        lambda_new = lambda0 + lambda1
    elif not diag0:
        lambda_new = lambda0 + torch.diag_embed(lambda1).float()
    elif not diag1:
        lambda_new = torch.diag_embed(lambda0).float() + lambda1

    eta_new = eta0 + eta1

    if diag0 and diag1:
        sigma_new = 1.0 / lambda_new
        mu_new = torch.matmul(sigma_new, eta_new.unsqueeze(-1)).squeeze(-1)
    else:
        sigma_new = torch.inverse(lambda_new)
        mu_new = torch.matmul(sigma_new, eta_new.unsqueeze(-1)).squeeze(-1)

    if need_zeta:
        if diag0:
            zeta0 = fast_calculate_zeta(eta0, lambda0, mu=mu0)
        else:
            zeta0 = calculate_zeta(eta0, lambda0, mu=mu0)

        if diag1:
            zeta1 = fast_calculate_zeta(eta1, lambda1, mu=mu1)
        else:
            zeta1 = calculate_zeta(eta1, lambda1, mu=mu1)

        if diag0 and diag1:
            zeta_new = fast_calculate_zeta(eta_new, lambda_new, mu=mu_new)
        else:
            zeta_new = calculate_zeta(eta_new, lambda_new, mu=mu_new)

        score = zeta0 + zeta1 - zeta_new
    else:
        score = None
    return score, mu_new, sigma_new


# if forward, integral the upper half. Else, lower half.
# the first Gaussian is regarded as the transition score.
# the second Gaussian is regarded as the emission/inside score.
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

    select = 1 if forward else 0

    res_mu = torch.split(mu_new, split_size_or_sections=[dim1, dim1], dim=-1)[select]
    res_sigma = torch.split(torch.split(sigma_new, split_size_or_sections=[dim1, dim1], dim=-2)[select],
                            split_size_or_sections=[dim1, dim1], dim=-1)[select]

    return scale, res_mu, res_sigma


# here mu1 is diagonal format
# TODO store fixed gaussian to reuse here.
def fast_gaussian_multi_integral(mu0: torch.Tensor, mu1: torch.Tensor,
                                 var0: torch.Tensor, var1: torch.Tensor,
                                 forward: bool = True, need_zeta: bool = True,
                                 diag1: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    if diag1:
        lambda1 = torch.diag_embed(1.0 / var1).float()
    else:
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

    select = 1 if forward else 0

    res_mu = torch.split(mu_new, split_size_or_sections=[dim1, dim1], dim=-1)[select]
    res_sigma = torch.split(torch.split(sigma_new, split_size_or_sections=[dim1, dim1], dim=-2)[select],
                            split_size_or_sections=[dim1, dim1], dim=-1)[select]

    return scale, res_mu, res_sigma


# here we only consider score in shape [batch, component].
# If current component less than max component, we full it with normal gaussian
def gaussian_top_k_pruning(score, mu, var, k=1):
    batch, _, dim = mu.size()
    device = score.device
    if score.size()[-1] == k:
        return score, mu, var
    elif score.size()[-1] < k:
        gap = k - score.size()[-1]
        gap_score = (torch.min(score).detach() - 1e6).expand(batch, gap).to(device)
        gap_mu = torch.zeros(batch, gap, dim).to(device)
        gap_var = torch.eye(dim).repeat(batch, gap, 1, 1).to(device)
        return torch.cat([score, gap_score], dim=1), torch.cat([mu, gap_mu], dim=1), torch.cat([var, gap_var], dim=1)
    pruned_score, index = torch.topk(score, k, dim=-1)
    pruned_mu = torch.gather(mu, 1, index.view(batch, -1, 1).repeat(1, 1, dim))
    pruned_var = torch.gather(var, 1, index.view(batch, -1, 1, 1).repeat(1, 1, dim, dim))
    return pruned_score, pruned_mu, pruned_var

# here we consider pruning mixture gaussian by score.
# alert here has not batch
# If the score of current gaussian is smaller than max_score + k, pruning it
# max component should be 100
def gaussian_max_k_pruning(score, mu, var, k=-2.3):
    dim = mu.size()[-1]
    max_score, max_index = torch.max(score, dim=0)
    threshold_score = max_score + k
    mask = torch.gt(score, threshold_score)
    pruned_score = torch.masked_select(score, mask)
    pruned_mu = torch.masked_select(mu, mask.view(-1, 1).repeat(1, dim)).view(-1, dim)
    pruned_var = torch.masked_select(var, mask.view(-1, 1, 1).repeat(1, dim, dim)).view(-1, dim, dim)

    if pruned_score.size()[0] > 100:
        pruned_score_v2, index = torch.topk(pruned_score, 100, dim=-1)
        pruned_mu_v2 = torch.gather(pruned_mu, 0, index.view(-1, 1).repeat(1, dim))
        pruned_var_v2 = torch.gather(pruned_var, 0, index.view(-1, 1, 1).repeat(1, dim, dim))
        return pruned_score_v2, pruned_mu_v2, pruned_var_v2
    else:
        return pruned_score, pruned_mu, pruned_var

