from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter


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
        part2 = 2.0 * torch.sum(torch.log(torch.abs(torch.diagonal(torch.cholesky(lam),
                                                                   dim1=-1, dim2=-2))), dim=-1)

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
                   var0: torch.Tensor, var1: torch.Tensor) -> Tuple[torch.Tensor,
                                                                    torch.Tensor, torch.Tensor]:
    lambda0 = torch.inverse(var0)
    lambda1 = torch.inverse(var1)

    eta0 = torch.matmul(lambda0, mu0.unsqueeze(-1)).squeeze(-1)
    eta1 = torch.matmul(lambda1, mu1.unsqueeze(-1)).squeeze(-1)

    lambda_new = lambda0 + lambda1
    eta_new = eta0 + eta1
    sigma_new = torch.inverse(lambda_new)
    mu_new = torch.matmul(sigma_new, eta_new.unsqueeze(-1)).squeeze()
    zeta0 = calculate_zeta(eta0, lambda0, mu=mu0)
    zeta1 = calculate_zeta(eta1, lambda1, mu=mu1)
    zeta_new = calculate_zeta(eta_new, lambda_new, mu=mu_new)

    score = zeta0 + zeta1 - zeta_new
    return score, mu_new, sigma_new


# if forward, integral the upper half. Else, lower half.
# the first Gaussian is regarded as the emission/inside score.
# the second Gaussian is regarded as the transition score.
def gaussian_multi_integral(mu0: torch.Tensor, mu1: torch.Tensor,
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

    # here may not need zeta
    zeta0 = calculate_zeta(eta0.unsqueeze(-1), lambda0, mu=mu0_new)
    zeta1 = calculate_zeta(eta1.unsqueeze(-1), lambda1, mu=mu1_new)
    zeta_new = calculate_zeta(eta_new, lambda_new, sig=sigma_new)

    scale = zeta0 + zeta1 - zeta_new

    if forward:
        res_mu = mu_new[:, dim1:]
        res_sigma = sigma_new[:, dim1:, dim1:]
    else:
        res_mu = mu_new[:, dim1]
        res_sigma = sigma_new[:, dim1, :dim1]

    return scale, res_mu, res_sigma


def reset_embedding(init_embedding, embedding_layer, embedding_dim, trainable):
    if init_embedding is None:
        # here random init the mu can be seen as the normal embedding
        # but for the variance, maybe we should use other method to init it.
        # Currently, the init only works for mu.
        scale = np.sqrt(3.0 / embedding_dim)
        embedding_layer.weight.data.uniform_(-scale, scale)
    else:
        embedding_layer.load_state_dict({'weight': init_embedding})
    embedding_layer.weight.requires_grad = trainable


class LanguageModel(nn.Module):

    def __init__(self):
        super(LanguageModel, self).__init__()

    def get_loss(self, sentences, masks):
        decoded = self.forward(sentences, masks)
        sentences = (sentences[:, 1:]).contiguous()
        masks = (masks[:, 1:]).contiguous()
        decoded = (decoded[:, :-1]).contiguous()
        return torch.sum(
            self.criterion(decoded.view(-1, self.ntoken), sentences.view(-1)).view(
                sentences.size()) * masks) / torch.sum(masks)

    def inference(self, sentences: torch.Tensor, masks: torch.Tensor):
        decoded = self.forward(sentences, masks)
        golder_sentences = (sentences[:, 1:]).numpy()
        masks = (masks[:, 1:]).numpy()
        decoded = (decoded[:, :-1]).contiguous()
        predict_sentence = torch.argmax(decoded, dim=-1).numpy()
        correct_number = np.sum(np.equal(predict_sentence, golder_sentences) * masks)
        correct_acc = correct_number / np.sum(masks)
        return predict_sentence * masks, correct_number, correct_acc


class GaussianBatchLanguageModel(LanguageModel):
    def __init__(self, dim: int, vocab_size: int, mu_embedding=None, var_embedding=None, init_var_scale=1.0):
        super(GaussianBatchLanguageModel, self).__init__()
        self.dim = dim
        self.ntoken = vocab_size
        self.emission_mu_embedding = nn.Embedding(vocab_size, self.dim)
        self.emission_cho_embedding = nn.Embedding(vocab_size, self.dim)

        self.transition_mu = Parameter(torch.empty(2 * self.dim), requires_grad=True)
        self.transition_cho = Parameter(torch.empty(2 * self.dim, 2 * self.dim), requires_grad=True)

        self.decoder_mu = Parameter(torch.empty(vocab_size, self.dim), requires_grad=True)
        self.decoder_cho = Parameter(torch.empty(vocab_size, self.dim), requires_grad=True)

        self.criterion = nn.CrossEntropyLoss(reduction='none')

        reset_embedding(mu_embedding, self.emission_mu_embedding, self.dim, True)
        reset_embedding(var_embedding, self.emission_cho_embedding, self.dim, True)

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

    def forward(self, sentences: torch.Tensor, masks: torch.Tensor):
        batch, max_len = sentences.size()
        swapped_sentences = sentences.transpose(0, 1)
        # update transition cho to variance
        trans_var = atma(self.transition_cho)
        # update cho_embedding to var_embedding
        word_mu_mat = self.emission_mu_embedding(swapped_sentences)
        word_var_embedding = self.emission_cho_embedding(swapped_sentences) ** 2
        word_var_mat = torch.diag_embed(word_var_embedding).float()

        decoder_var = torch.diag_embed(self.decoder_cho ** 2).float()

        holder_mu = []
        holder_var = []
        trans_mu = self.transition_mu
        # init
        inside_mu = torch.zeros(batch, self.dim, requires_grad=False)
        inside_var = torch.eye(self.dim, requires_grad=False).repeat(batch, 1, 1)
        for i in range(max_len):
            # update inside score from pred and current inside score
            _, part_mu, part_var = gaussian_multi_integral(trans_mu, inside_mu, trans_var, inside_var)
            _, inside_mu, inside_var = gaussian_multi(part_mu, word_mu_mat[i], part_var, word_var_mat[i])
            holder_mu.append(inside_mu)
            holder_var.append(inside_var)

        # get right word, calculate
        # pred_mu shape [len, dim]
        # pred_var shape [len, dim, dim]
        pred_mus = torch.stack(holder_mu, dim=1)
        pred_vars = torch.stack(holder_var, dim=1)
        score, _, _ = gaussian_multi(pred_mus.unsqueeze(2), self.decoder_mu.view((1, 1) + self.decoder_mu.size()),
                                     pred_vars.unsqueeze(2), decoder_var.view((1, 1) + decoder_var.size()))

        # shape [batch, len-1, vocab_size]
        real_score = score.squeeze()
        return real_score

    # def evaluate(self, sentence):
    #     loss = self.forward(sentence)
    #     len = sentence.size(0)
    #     return loss.item() * len
    #
    # def get_loss(self, sentences, masks):
    #     decoded = self.forward(sentences)
    #     sentences = (sentences[:, 1:]).contiguous()
    #     masks = (masks[:, 1:]).contiguous()
    #     decoded = (decoded[:, :-1]).contiguous()
    #     return torch.sum(
    #         self.criterion(decoded.view(-1, self.ntoken), sentences.view(-1)).view(
    #             sentences.size()) * masks) / torch.sum(masks)
    #
    # def inference(self, sentences: torch.Tensor, masks: torch.Tensor):
    #     decoded = self.forward(sentences)
    #     golder_sentences = (sentences[:, 1:]).numpy()
    #     masks = (masks[:, 1:]).numpy()
    #     decoded = (decoded[:, :-1]).contiguous()
    #     predict_sentence = torch.argmax(decoded, dim=-1).numpy()
    #     correct_number = np.sum(np.equal(predict_sentence, golder_sentences) * masks)
    #     correct_acc = correct_number / np.sum(masks)
    #     return predict_sentence * masks, correct_number, correct_acc


class RNNLanguageModel(LanguageModel):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers=1, dropout=0.0, tie_weights=False):
        super(RNNLanguageModel, self).__init__()
        self.ntoken = ntoken + 2
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(self.ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(nhid, self.ntoken)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, sentences: torch.Tensor, masks: torch.Tensor):
        emb = self.drop(self.encoder(sentences))
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, masks.sum(dim=1), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_emb)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        decoded = self.decoder(self.drop(output))
        return decoded

    # def get_loss(self, sentences: torch.Tensor, masks: torch.Tensor):
    #     decoded = self.forward(sentences, masks)
    #     # rename after
    #     sentences = (sentences[:, 1:]).contiguous()
    #     masks = (masks[:, 1:]).contiguous()
    #     decoded = (decoded[:, :-1]).contiguous()
    #     return torch.sum(
    #         self.criterion(decoded.view(-1, self.ntoken), sentences.view(-1)).view(
    #             sentences.size()) * masks) / torch.sum(masks)
    #
    # def inference(self, sentences: torch.Tensor, masks: torch.Tensor):
    #     decoded = self.forward(sentences, masks)
    #     golder_sentences = (sentences[:, 1:]).numpy()
    #     masks = (masks[:, 1:]).numpy()
    #     decoded = (decoded[:, :-1]).contiguous()
    #     predict_sentence = torch.argmax(decoded, dim=-1).numpy()
    #     correct_number = np.sum(np.equal(predict_sentence, golder_sentences) * masks)
    #     correct_acc = correct_number / np.sum(masks)
    #     return predict_sentence * masks, correct_number, correct_acc
