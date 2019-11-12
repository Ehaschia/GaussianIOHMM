import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
# in order to use the global variable
import global_variables
from model.gaussian_basic_opration import *


def reset_embedding(init_embedding, embedding_layer, embedding_dim, trainable, far_init):
    if init_embedding is None:
        # here random init the mu can be seen as the normal embedding
        # but for the variance, maybe we should use other method to init it.
        # Currently, the init only works for mu.
        if far_init:
            scale = 1.0
        else:
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
            self.criterion(decoded.view(-1, self.ntokens), sentences.view(-1)).view(
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
    def __init__(self, dim: int, ntokens: int, mu_embedding=None, var_embedding=None, init_var_scale=1.0):
        super(GaussianBatchLanguageModel, self).__init__()
        self.dim = dim
        self.ntokens = ntokens + 2
        self.emission_mu_embedding = nn.Embedding(self.ntokens, self.dim)
        self.emission_cho_embedding = nn.Embedding(self.ntokens, self.dim)

        self.transition_mu = Parameter(torch.empty(2 * self.dim), requires_grad=True)
        self.transition_cho = Parameter(
            torch.empty(2 * self.dim, 2 * self.dim), requires_grad=global_variables.TRANSITION_CHO_GRAD)

        self.decoder_mu = Parameter(torch.empty(self.ntokens, self.dim), requires_grad=True)
        self.decoder_cho = Parameter(
            torch.empty(self.ntokens, self.dim), requires_grad=global_variables.DECODE_CHO_GRAD)

        self.criterion = nn.CrossEntropyLoss(reduction='none')

        reset_embedding(mu_embedding, self.emission_mu_embedding, self.dim, True,
                        far_init=global_variables.FAR_EMISSION_MU)
        reset_embedding(var_embedding, self.emission_cho_embedding, self.dim, global_variables.EMISSION_CHO_GRAD,
                        far_init=False)
        self.reset_parameter(init_var_scale)

    def reset_parameter(self, init_var_scale):
        if global_variables.FAR_TRANSITION_MU:
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
        if global_variables.FAR_DECODE_MU:
            nn.init.uniform_(self.decoder_mu, a=-1.0, b=1.0)
        else:
            nn.init.xavier_normal_(self.decoder_mu)
        nn.init.uniform_(self.decoder_cho)

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
            _, part_mu, part_var = gaussian_multi_integral(trans_mu, inside_mu, trans_var, inside_var, need_zeta=False)
            _, inside_mu, inside_var = gaussian_multi(part_mu, word_mu_mat[i], part_var, word_var_mat[i], need_zeta=False)
            holder_mu.append(inside_mu)
            holder_var.append(inside_var)

        # get right word, calculate
        # pred_mu shape [batch, len, dim]
        # pred_var shape [batch, len, dim, dim]
        pred_mus = torch.stack(holder_mu, dim=1)
        pred_vars = torch.stack(holder_var, dim=1)
        score, _, _ = gaussian_multi(pred_mus.unsqueeze(2),
                                     self.decoder_mu.view((1, 1) + self.decoder_mu.size()),
                                     pred_vars.unsqueeze(2),
                                     decoder_var.view((1, 1) + decoder_var.size()),
                                     need_zeta=True)

        # shape [batch, len-1, vocab_size]
        real_score = score.squeeze()
        return real_score


class RNNLanguageModel(LanguageModel):
    def __init__(self, rnn_type, ntokens, ninp, nhid, nlayers=1, dropout=0.0, tie_weights=False):
        super(RNNLanguageModel, self).__init__()
        self.ntokens = ntokens + 2
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
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(nhid, self.ntokens)
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
