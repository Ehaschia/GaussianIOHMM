from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter


class BasicFramework(nn.Module):
    def __init__(self, dim: int):
        super(BasicFramework, self).__init__()
        self.dim = dim

    # A^T multiply A
    def atma(self, cholesky: torch.Tensor) -> torch.Tensor:
        return torch.matmul(cholesky.transpose(-2, -1), cholesky)

    def calculate_zeta(self, eta: torch.Tensor, lam: torch.Tensor,
                       mu: torch.Tensor = None, sig: torch.Tensor = None) -> torch.Tensor:
        # zeta in log fromat
        # zeta = - 0.5 * (dim * log 2pi  - log|lam| + eta^T * lam^{-1} * eta)
        # zeta = - 0.5 * (dim * log 2pi  + log|sigma| + eta^T * sigma * eta)
        # TODO optimize calculate process, such as determinate and inverse

        dim = lam.size(-1)
        part1 = dim * np.log(np.pi * 2)
        if len(lam.size()) == 2:
            part2 = torch.log(torch.det(lam))
        else:
            part2 = 2.0 * torch.sum(torch.log(torch.abs(torch.diagonal(torch.cholesky(lam),
                                                                       dim1=-1, dim2=-2))), dim=-1)

        if sig is None: # and mu is None:
            sig = torch.inverse(lam)

        if mu is not None:
            if len(eta.size()) > 1:
                part3 = torch.matmul(eta.transpose(-2, -1), mu).reshape(eta.size()[:-2])
            else:
                part3 = torch.matmul(eta, torch.matmul(sig, eta))
        else:
            part3 = torch.matmul(eta, torch.matmul(sig, eta))

        return -0.5 * (part1 - part2 + part3)

    def gaussian_multi(self, mu0: torch.Tensor, mu1: torch.Tensor,
                       var0: torch.Tensor, var1: torch.Tensor) -> Tuple[torch.Tensor,
                                                                        torch.Tensor, torch.Tensor]:
        lambda0 = torch.inverse(var0)
        lambda1 = torch.inverse(var1)

        if len(mu0.size()) > 1:
            mu0.unsqueeze_(-1)
        if len(mu1.size()) > 1:
            mu1.unsqueeze_(-1)
        eta0 = torch.matmul(lambda0, mu0)
        eta1 = torch.matmul(lambda1, mu1)

        lambda_new = lambda0 + lambda1
        eta_new = eta0 + eta1
        sigma_new = torch.inverse(lambda_new)
        mu_new = torch.matmul(sigma_new, eta_new)
        zeta0 = self.calculate_zeta(eta0, lambda0, mu=mu0)
        zeta1 = self.calculate_zeta(eta1, lambda1, mu=mu1)
        zeta_new = self.calculate_zeta(eta_new, lambda_new, mu=mu_new)

        score = zeta0 + zeta1 - zeta_new
        return score, mu_new, sigma_new

    def gaussian_multi_integral(self, mu0: torch.Tensor, mu1: torch.Tensor,
                                var0: torch.Tensor, var1: torch.Tensor,
                                forward: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

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

        # use new avoid leaf variable in-place opreation
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
        mu_new = torch.matmul(sigma_new, eta_new)

        # here may not need zeta
        zeta0 = self.calculate_zeta(eta0, lambda0, mu=mu0_new)
        zeta1 = self.calculate_zeta(eta1, lambda1, mu=mu1_new)
        zeta_new = self.calculate_zeta(eta_new, lambda_new, sig=sigma_new)

        scale = zeta0 + zeta1 - zeta_new

        if forward:
            res_mu = mu_new[dim1:]
            res_sigma = sigma_new[dim1:, dim1:]
        else:
            res_mu = mu_new[:dim1]
            res_sigma = sigma_new[:dim1, :dim1]

        return scale, res_mu, res_sigma


class GaussianLanguageModel(BasicFramework):
    def __init__(self, dim: int, vocb_size: int, mu_embedding=None,
                 var_embedding=None, init_var_scale = 1.0):
        super(GaussianLanguageModel, self).__init__(dim)

        self.emission_mu_embedding = nn.Embedding(vocb_size, self.dim)
        self.emission_cho_embedding = nn.Embedding(vocb_size, self.dim)

        self.transition_mu = Parameter(torch.Tensor(2 * self.dim))
        self.transition_cho = Parameter(torch.Tensor(2 * self.dim, 2 * self.dim))

        self.decoder_mu = Parameter(torch.Tensor(vocb_size, self.dim))
        self.decoder_cho = Parameter(torch.Tensor(vocb_size, self.dim))
        self.criterion = nn.CrossEntropyLoss()
        self.reset_embedding(mu_embedding, self.emission_mu_embedding, self.dim, True)
        self.reset_embedding(var_embedding, self.emission_cho_embedding, self.dim, True)
        self.reset_parameter(init_var_scale)

    def reset_parameter(self, init_var_scale):
        nn.init.uniform_(self.transition_mu)
        # here the init of var should be alert
        nn.init.uniform_(self.transition_cho)
        weight = self.transition_cho.data - 0.5
        # maybe here we need to add some 
        weight = torch.tril(weight)
        # weight = self.atma(weight)
        self.transition_cho.data = weight + init_var_scale * torch.eye(2 * self.dim)

        nn.init.xavier_normal_(self.decoder_mu)
        nn.init.xavier_normal_(self.decoder_cho)

        # var_weight = self.decoder_var.data
        # var_weight = var_weight * var_weight
        # self.decoder_var.data = var_weight
        # var_weight = self.emission_var_embedding.weight.data
        # self.emission_var_embedding.weight.data = var_weight * var_weight

    def reset_embedding(self, init_embedding, embedding_layer, embedding_dim, trainable):
        if init_embedding is None:
            # here random init the mu can be seen as the noraml embedding
            # but for the variance, maybe we should use other method to init it.
            # ALERT the init only work for mu
            scale = np.sqrt(3.0 / embedding_dim)
            embedding_layer.weight.data.uniform_(-scale, scale)
        else:
            embedding_layer.load_state_dict({'weight': init_embedding})
        embedding_layer.weight.requires_grad = trainable

    def forward(self, sentence):
        len = sentence.size()[0]

        # update transition cho to var cho
        trans_var = self.atma(self.transition_cho)
        # update cho_embedding to var_embedding
        word_mu_mat = self.emission_mu_embedding(sentence)
        word_var_embedding = self.emission_cho_embedding(sentence) ** 2
        word_var_mat = torch.diag_embed(word_var_embedding).float()

        decoder_var = torch.diag_embed(self.decoder_cho ** 2).float()

        holder_mu = []
        holder_var = []
        trans_mu = self.transition_mu
        inside_mu, inside_var = None, None
        for i in range(len-1):
            # only at position 0, inside mu is None, here inside is emission of current work
            if inside_mu is None:
                inside_mu = word_mu_mat[i]
                inside_var = word_var_mat[i]
            else:
                # update inside score from pred and current inside score
                _, inside_mu, inside_var = self.gaussian_multi(pred_mu, word_mu_mat[i], pred_var, word_var_mat[i])
            _, pred_mu, pred_var = self.gaussian_multi_integral(trans_mu, inside_mu,
                                                                  trans_var, inside_var)

            holder_mu.append(pred_mu)
            holder_var.append(pred_var)

            # get right word, calculate
        # pred_mu shape [len, dim]
        # pred_var shape [len, dim, dim]
        pred_mus = torch.stack(holder_mu)
        pred_vars = torch.stack(holder_var)
        score, _, _ = self.gaussian_multi(pred_mus.unsqueeze(0), self.decoder_mu.unsqueeze(1),
                                          pred_vars.unsqueeze(0), decoder_var.unsqueeze(1))

        # shape [batch, len-1, vocab_size]
        real_score = score.squeeze()
        target = sentence[1:]
        loss = self.criterion(real_score.transpose(-2, -1), target)
        return loss

    def evaluate(self, sentence):
        loss = self.forward(sentence)
        len = sentence.size(0)
        return loss.item() * len
