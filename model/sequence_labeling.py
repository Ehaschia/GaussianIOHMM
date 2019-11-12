from model.gaussian_basic_opration import *
import torch.nn as nn
from torch.nn import Parameter
from global_variables import *


def reset_embedding(init_embedding, embedding_layer, embedding_dim, trainable, far_init):
    if init_embedding is None:
        if far_init:
            scale = 1.0
        else:
            scale = np.sqrt(3.0 / embedding_dim)
        embedding_layer.weight.data.uniform_(-scale, scale)
    else:
        embedding_layer.load_state_dict({'weight': init_embedding})
    embedding_layer.weight.requires_grad = trainable


class GaussianSequenceLabeling(nn.Module):

    def __init__(self, dim: int, ntokens: int, nlabels: int, mu_embedding=None, var_embedding=None, init_var_scale=1.0):
        super(GaussianSequenceLabeling, self).__init__()
        # emission
        self.dim = dim
        self.ntokens = ntokens + 2
        self.nlabels = nlabels
        # parameter init
        self.emission_mu_embedding = nn.Embedding(self.ntokens, self.dim)
        self.emission_cho_embedding = nn.Embedding(self.ntokens, self.dim)

        self.transition_mu = Parameter(torch.empty(2 * self.dim), requires_grad=True)
        self.transition_cho = Parameter(
            torch.empty(2 * self.dim, 2 * self.dim), requires_grad=TRANSITION_CHO_GRAD)

        self.decoder_mu = Parameter(torch.empty(self.nlabels, self.dim), requires_grad=True)
        self.decoder_cho = Parameter(
            torch.empty(self.nlabels, self.dim), requires_grad=DECODE_CHO_GRAD)

        self.criterion = nn.CrossEntropyLoss(reduction='none')

        reset_embedding(mu_embedding, self.emission_mu_embedding, self.dim, True,
                        far_init=FAR_EMISSION_MU)
        reset_embedding(var_embedding, self.emission_cho_embedding, self.dim, EMISSION_CHO_GRAD,
                        far_init=False)
        self.reset_parameter(init_var_scale)

    def reset_parameter(self, init_var_scale):
        if FAR_TRANSITION_MU:
            nn.init.uniform_(self.transition_mu, a=-1.0, b=1.0)
        else:
            to_init_transition_mu = self.transition_mu.unsqueeze(0)
            nn.init.xavier_normal_(to_init_transition_mu)
            self.transition_mu.data = to_init_transition_mu.squeeze(0)
        # here the init of var should be alert
        nn.init.uniform_(self.transition_cho)
        weight = self.transition_cho.data - 0.5
        # maybe here we need to add some
        weight = torch.tril(weight)
        # weight = self.atma(weight)
        self.transition_cho.data = weight + init_var_scale * torch.eye(2 * self.dim)
        if FAR_DECODE_MU:
            nn.init.uniform_(self.decoder_mu, a=-1.0, b=1.0)
        else:
            nn.init.xavier_normal_(self.decoder_mu)
        nn.init.uniform_(self.decoder_cho)

    def forward(self, sentences: torch.Tensor, backward_order: torch.Tensor) -> torch.Tensor:
        batch, max_len = sentences.size()
        swapped_sentences = sentences.transpose(0, 1)
        # update transition cho to variance
        trans_var = atma(self.transition_cho)
        # update cho_embedding to var_embedding
        word_mu_mat = self.emission_mu_embedding(swapped_sentences)
        word_var_embedding = self.emission_cho_embedding(swapped_sentences) ** 2
        word_var_mat = torch.diag_embed(word_var_embedding).float()

        decoder_var = torch.diag_embed(self.decoder_cho ** 2).float()

        trans_mu = self.transition_mu
        # init
        init_mu = torch.zeros(batch, self.dim, requires_grad=False)
        init_var = torch.eye(self.dim, requires_grad=False).repeat(batch, 1, 1)
        forward_mu = init_mu
        forward_var = init_var
        forward_holder_mu = [init_mu]
        forward_holder_var = [init_var]
        # forward
        for i in range(max_len):
            # update forward score from pred and current inside score
            _, part_mu, part_var = gaussian_multi_integral(trans_mu, forward_mu, trans_var, forward_var, forward=True)
            _, forward_mu, forward_var = gaussian_multi(part_mu, word_mu_mat[i], part_var, word_var_mat[i])
            forward_holder_mu.append(forward_mu)
            forward_holder_var.append(forward_var)

        # get right word, calculate
        # pred_mu shape [batch, len, dim]
        # pred_var shape [batch, len, dim, dim]
        forward_mus = torch.stack(forward_holder_mu, dim=0)
        forward_vars = torch.stack(forward_holder_var, dim=0)

        # backward
        backward_holder_mu = []
        backward_holder_var = []
        backward_mu = init_mu
        backward_var = init_var
        for i in range(max_len):
            _, part_mu, part_var = gaussian_multi_integral(trans_mu, backward_mu, trans_var, backward_var,
                                                           forward=False, need_zeta=False)
            # TODO debug here
            _, backward_mu, backward_var = gaussian_multi(
                part_mu,
                torch.gather(word_mu_mat, 0,
                             backward_order[:, i].view(batch, 1).repeat(1, self.dim).unsqueeze(0)).squeeze(),
                part_var,
                torch.gather(word_var_mat, 0,
                             backward_order[:, i].view(batch, 1, 1).repeat(1, self.dim, self.dim).unsqueeze(
                                 0)).squeeze(),
                need_zeta=False)
            backward_holder_mu.append(backward_mu)
            backward_holder_var.append(backward_var)
        backward_holder_mu.append(init_mu)
        backward_holder_var.append(init_var)
        backward_mus = torch.stack(backward_holder_mu, dim=0)
        backward_vars = torch.stack(backward_holder_var, dim=0)

        expected_count_mu_holder = []
        expected_count_var_holder = []
        # here calculate the expected count at position i.
        # we need to count previous position forward score i-1
        # and next position backward score i+1. Thus we here begin with 1.
        for i in range(0, max_len):
            prev_forward_mu = forward_mus[i]
            prev_forward_var = forward_vars[i]
            current_emission_mu = word_mu_mat[i]
            current_emission_var = word_var_mat[i]
            if i + 1 != max_len:
                backward_idx = torch.where(backward_order == i + 1)[1]
                next_backward_mu = torch.gather(
                    backward_mus, 0, backward_idx.view(batch, 1).repeat(1, self.dim).unsqueeze(0)).squeeze()
                next_backward_var = torch.gather(backward_vars, 0,
                                                 backward_idx.view(batch, 1, 1).repeat(1, self.dim, self.dim).unsqueeze(
                                                     0)).squeeze()
            else:
                next_backward_mu = backward_mus[i + 1]
                next_backward_var = backward_vars[i + 1]
            _, part_mu, part_var = gaussian_multi(prev_forward_mu, current_emission_mu, prev_forward_var,
                                                  current_emission_var)
            _, expected_count_mu, expected_count_var = gaussian_multi(part_mu, next_backward_mu, part_var,
                                                                      next_backward_var)
            expected_count_mu_holder.append(expected_count_mu)
            expected_count_var_holder.append(expected_count_var)

        expected_count_mus = torch.stack(expected_count_mu_holder, dim=1)
        expected_count_vars = torch.stack(expected_count_var_holder, dim=1)
        score, _, _ = gaussian_multi(expected_count_mus.unsqueeze(2),
                                     self.decoder_mu.view((1, 1) + self.decoder_mu.size()),
                                     expected_count_vars.unsqueeze(2), decoder_var.view((1, 1) + decoder_var.size()))

        # shape [batch, len-1, vocab_size]
        real_score = score.squeeze()
        return real_score

    def get_loss(self, sentences: torch.Tensor, labels: torch.Tensor,
                 masks: torch.Tensor, backward_order: torch.Tensor) -> torch.Tensor:
        real_score = self.forward(sentences, backward_order)
        prob = self.criterion(real_score.view(-1, self.nlabels), labels.view(-1)).reshape_as(labels) * masks
        # TODO the loss format can fine tune. According to zechuan's investigation.
        return torch.sum(prob) / sentences.size(0)

    def get_acc(self, sentences: torch.Tensor, labels: torch.Tensor,
                masks: torch.Tensor, backward_order: torch.Tensor) -> Tuple:
        real_score = self.forward(sentences, backward_order)
        pred = torch.argmax(real_score, dim=-1).cpu().numpy()
        corr = np.sum(np.equal(pred, labels.cpu().numpy()) * masks.cpu().numpy())
        total = np.sum(masks.cpu().numpy())
        return corr / total, corr


class RNNSequenceLabeling(nn.Module):
    def __init__(self, rnn_type, ntokens, nlabels, ninp, nhid, nlayers=1, dropout=0.0, tie_weights=False):
        super(RNNSequenceLabeling, self).__init__()
        self.ntokens = ntokens + 2
        self.nlabels = nlabels
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(self.ntokens, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout, batch_first=True,
                              bidirectional=True)
        self.decoder = nn.Linear(2 * nhid, self.nlabels)
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
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, masks.sum(dim=1)+2, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_emb)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        decoded = self.decoder(self.drop(output))
        return decoded

    def get_loss(self, sentences: torch.Tensor, labels: torch.Tensor,
                 masks: torch.Tensor) -> torch.Tensor:
        real_score = self.forward(sentences, masks)
        prob = self.criterion(real_score.view(-1, self.nlabels), labels.view(-1)).reshape_as(labels) * masks
        # TODO the loss format can fine tune. According to zechuan's investigation.
        return torch.sum(prob) / sentences.size(0)

    def get_acc(self, sentences: torch.Tensor, labels: torch.Tensor,
                masks: torch.Tensor) -> Tuple:
        real_score = self.forward(sentences, masks)
        pred = torch.argmax(real_score, dim=-1).cpu().numpy()
        corr = np.sum(np.equal(pred, labels.cpu().numpy()) * masks.cpu().numpy())
        total = np.sum(masks.cpu().numpy())
        return corr / total, corr