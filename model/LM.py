import torch.nn as nn
from torch.nn import Parameter
# in order to use the global variable
from global_variables import *
from model.basic_operation import *
from scipy.stats import invwishart
from model.utils import *

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
            torch.empty(2 * self.dim, 2 * self.dim), requires_grad=TRANSITION_CHO_GRAD)

        self.decoder_mu = Parameter(torch.empty(self.ntokens, self.dim), requires_grad=True)
        self.decoder_cho = Parameter(
            torch.empty(self.ntokens, self.dim), requires_grad=DECODE_CHO_GRAD)

        self.criterion = nn.CrossEntropyLoss(reduction='none')

        reset_embedding(mu_embedding, self.emission_mu_embedding, self.dim, True)
        reset_embedding(var_embedding, self.emission_cho_embedding, self.dim, EMISSION_CHO_GRAD)
        self.reset_parameter(init_var_scale)

    def reset_parameter(self, init_var_scale):
        to_init_transition_mu = self.transition_mu.unsqueeze(0)
        nn.init.xavier_normal_(to_init_transition_mu)
        self.transition_mu.data = to_init_transition_mu.squeeze(0)
        # here the init of transition var should be alert
        nn.init.uniform_(self.transition_cho)
        weight = self.transition_cho.data - 0.5
        # maybe here we need to add some
        weight = torch.tril(weight)
        self.transition_cho.data = weight + init_var_scale * torch.eye(2 * self.dim)

        nn.init.xavier_normal_(self.decoder_mu)
        nn.init.uniform_(self.decoder_cho)

        # var_weight = self.decoder_var.data
        # var_weight = var_weight * var_weight
        # self.decoder_var.data = var_weight
        # var_weight = self.emission_var_embedding.weight.data
        # self.emission_var_embedding.weight.data = var_weight * var_weight

    def forward(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
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
            _, inside_mu, inside_var = gaussian_multi(part_mu, word_mu_mat[i], part_var, word_var_mat[i],
                                                      need_zeta=False)
            holder_mu.append(inside_mu)
            holder_var.append(inside_var)

        # get right word, calculate
        # pred_mu shape [batch, len, dim]
        # pred_var shape [batch, len, dim, dim]
        pred_mus = torch.stack(holder_mu, dim=1)
        pred_vars = torch.stack(holder_var, dim=1)
        # result should be [batch_size, length, s]
        score, _, _ = gaussian_multi(pred_mus.unsqueeze(2),
                                     self.decoder_mu.view((1, 1) + self.decoder_mu.size()),
                                     pred_vars.unsqueeze(2),
                                     decoder_var.view((1, 1) + decoder_var.size()),
                                     need_zeta=True)

        # shape [batch, len-1, vocab_size]
        real_score = score.squeeze()
        return real_score


class MixtureGaussianBatchLanguageModel(LanguageModel):
    def __init__(self, dim: int, ntokens: int, mu_embedding=None,
                 var_embedding=None, init_var_scale=1.0, i_comp_num=1,
                 t_comp_num=1, o_comp_num=1, max_comp=10):
        super(MixtureGaussianBatchLanguageModel, self).__init__()
        self.dim = dim
        self.ntokens = ntokens + 2
        self.max_comp = max_comp
        self.i_comp_num = i_comp_num
        self.t_comp_num = t_comp_num
        self.o_comp_num = o_comp_num
        self.input_mu_embedding = nn.Embedding(self.ntokens, i_comp_num * self.dim)
        self.input_cho_embedding = nn.Embedding(self.ntokens, i_comp_num * self.dim)

        self.transition_mu = Parameter(torch.empty(t_comp_num, 2 * self.dim), requires_grad=True)
        self.transition_cho = Parameter(
            torch.empty(t_comp_num, 2 * self.dim, 2 * self.dim), requires_grad=TRANSITION_CHO_GRAD)

        self.output_mu = Parameter(torch.empty(self.ntokens, o_comp_num * self.dim), requires_grad=True)
        self.output_cho = Parameter(
            torch.empty(self.ntokens, o_comp_num, self.dim), requires_grad=DECODE_CHO_GRAD)

        self.criterion = nn.CrossEntropyLoss(reduction='none')

        reset_embedding(mu_embedding, self.input_mu_embedding, self.dim, True)
        reset_embedding(var_embedding, self.input_cho_embedding, self.dim, EMISSION_CHO_GRAD)
        self.reset_parameter(init_var_scale=init_var_scale)

    def reset_parameter(self, trans_var_init='raw', init_var_scale=1.0):

        to_init_transition_mu = self.transition_mu.unsqueeze(0)
        nn.init.xavier_normal_(to_init_transition_mu)
        self.transition_mu.data = to_init_transition_mu.squeeze(0)
        if trans_var_init == 'raw':
            # here the init of var should be alert
            nn.init.uniform_(self.transition_cho)
            weight = self.transition_cho.data - 0.5
            # maybe here we need to add some
            weight = torch.tril(weight)
            # weight = self.atma(weight)
            self.transition_cho.data = weight + init_var_scale * torch.eye(2 * self.dim)
        elif trans_var_init == 'wishart':
            transition_var = invwishart.rvs(self.dim, np.eye(self.dim) / self.dim,
                                            size=self.t_comp_num, random_state=None)
            self.transition_cho.data = torch.from_numpy(np.linalg.cholesky(transition_var))
        else:
            raise ValueError("Error transition init method")
        nn.init.xavier_normal_(self.output_mu)
        nn.init.uniform_(self.output_cho)

    # ALERT this method just used for unit test
    def rewrite_parameter(self, dim, ntokens, i_comp_num, t_comp_num, o_comp_num, max_comp,
                          input_mu, input_cho, trans_mu, trans_cho, out_mu, out_cho):
        self.dim = dim
        self.ntokens = ntokens
        self.i_comp_num = i_comp_num
        self.t_comp_num = t_comp_num
        self.o_comp_num = o_comp_num
        self.max_comp = max_comp
        self.input_mu_embedding = nn.Embedding(self.ntokens, self.i_comp_num * self.dim)
        self.input_mu_embedding.weight.data = input_mu
        self.input_cho_embedding = nn.Embedding(self.ntokens, self.i_comp_num * self.dim)
        self.input_cho_embedding.weight.data = input_cho
        self.transition_mu.data = trans_mu
        self.transition_cho.data = trans_cho
        self.output_mu = Parameter(torch.empty(self.ntokens, self.o_comp_num * self.dim), requires_grad=True)
        self.output_mu.data = out_mu
        self.output_cho = Parameter(torch.empty(self.ntokens, self.o_comp_num * self.dim), requires_grad=True)
        self.output_cho.data = out_cho

    def forward(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch, max_len = sentences.size()
        swapped_sentences = sentences.transpose(0, 1)

        # update cho_embedding to var_embedding
        input_mu_mat = self.input_mu_embedding(swapped_sentences).reshape(max_len, batch, self.i_comp_num, self.dim)
        input_var_embedding = (self.input_cho_embedding(swapped_sentences) ** 2).reshape(max_len, batch,
                                                                                         self.i_comp_num, self.dim)
        input_var = torch.diag_embed(input_var_embedding).float()

        output_var = torch.diag_embed(self.output_cho ** 2).float().reshape(self.ntokens, self.o_comp_num, self.dim,
                                                                            self.dim)
        holder_score = []
        holder_mu = []
        holder_var = []

        # update transition cho to variance
        trans_var = atma(self.transition_cho).reshape(1, 1, self.t_comp_num, 2 * self.dim, 2 * self.dim)
        trans_mu = self.transition_mu.reshape(1, 1, self.t_comp_num, 2 * self.dim)
        # init
        prev_score = torch.zeros(batch, 1, 1)
        prev_mu = torch.zeros(batch, 1, 1, self.dim, requires_grad=False)
        prev_var = torch.eye(self.dim, requires_grad=False).repeat(batch, 1, 1).unsqueeze(1).unsqueeze(1)
        for i in range(max_len):
            # prev inside score multi transfer
            part_score, part_mu, part_var = gaussian_multi_integral(trans_mu, prev_mu, trans_var, prev_var,
                                                                    need_zeta=True)
            # part part multi current input
            inside_score, inside_mu, inside_var = gaussian_multi(
                part_mu.reshape(batch, -1, 1, self.dim),
                input_mu_mat[i].reshape(batch, 1, self.i_comp_num, self.dim),
                part_var.reshape(batch, -1, 1, self.dim, self.dim),
                input_var[i].reshape(batch, 1, self.i_comp_num, self.dim, self.dim),
                need_zeta=True)

            real_inside_score = (part_score + prev_score).reshape(batch, -1, 1) + inside_score
            # pruning
            prev_score, prev_mu, prev_var = gaussian_top_k_pruning(real_inside_score.reshape(batch, -1),
                                                                   inside_mu.reshape(batch, -1, self.dim),
                                                                   inside_var.reshape(batch, -1, self.dim, self.dim),
                                                                   k=self.max_comp)

            holder_score.append(prev_score)
            holder_mu.append(prev_mu)
            holder_var.append(prev_var)

        # get right word, calculate
        # pred_scores shape [batch, len, comp]
        # pred_mus shape [batch, len, comp, dim]
        # pred_vars shape [batch, len, comp, dim, dim]
        pred_scores = torch.stack(holder_score, dim=1)
        pred_mus = torch.stack(holder_mu, dim=1)
        pred_vars = torch.stack(holder_var, dim=1)
        # result should be [batch_size, len, comp1, comp2]
        # here comp1 from pred part and comp2 from
        score, _, _ = gaussian_multi(pred_mus.view(batch, max_len, 1, -1, 1, self.dim),
                                     self.output_mu.view(1, 1, self.ntokens, 1, self.o_comp_num, self.dim),
                                     pred_vars.view(batch, max_len, 1, -1, 1, self.dim, self.dim),
                                     output_var.view(1, 1, self.ntokens, 1, self.o_comp_num, self.dim, self.dim),
                                     need_zeta=True)

        # shape [batch, len-1, vocab_size]
        real_score = torch.logsumexp((pred_scores.unsqueeze(-1).unsqueeze(2) +
                                      score).reshape(batch, max_len, self.ntokens, -1), dim=3)
        return real_score


class RNNLanguageModel(LanguageModel):
    def __init__(self, rnn_type, ntokens, ninp, nhid, nlayers=1, dropout=0.0, tie_weights=False):
        super(RNNLanguageModel, self).__init__()
        self.ntokens = ntokens
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

    def get_loss(self, sentences: torch.Tensor, masks: torch.Tensor):
        emb = self.drop(self.encoder(sentences))
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, masks.sum(dim=1), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_emb)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        decoded = self.decoder(self.drop(output))
        ppl = self.criterion(decoded.view(-1, self.ntokens), sentences.view(-1))
        return torch.sum(ppl) / sentences.size(0)


class HMMLanguageModel(LanguageModel):
    def __init__(self, vocab_size, num_state=10):
        super(HMMLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.num_state = num_state
        # self.input = nn.Embedding(self.vocab_size, self.num_state)
        self.input = Parameter(torch.empty(self.vocab_size, self.num_state), requires_grad=True)
        self.transition = Parameter(torch.empty(self.num_state, self.num_state), requires_grad=True)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.logsoftmax1 = nn.LogSoftmax(dim=-1)
        self.logsoftmax0 = nn.LogSoftmax(dim=0)

        # TODO try this
        # self.adaptivelogsm = nn.AdaptiveLogSoftmaxWithLoss()
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.uniform_(self.transition.data, a=-0.5, b=0.5)
        nn.init.uniform_(self.input.data, a=-0.5, b=0.5)

    @staticmethod
    def bmv_log_product(bm, bv):
        return torch.logsumexp(bm + bv.unsqueeze(-2), dim=-1)

    @staticmethod
    def bmm_log_product(bm1, bm2):
        return torch.logsumexp(bm1.unsqueeze(-1) + bm2.unsqueeze(-3), dim=-2)

    def smooth_parameters(self):
        self.input.data = smoothing(self.input.data)
        self.transition.data = smoothing(self.transition.data)
        self.output.data = smoothing(self.output.data)

    def forward(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch, maxlen = sentences.size()

        swapped_sentence = sentences.transpose(0, 1)
        norm_input = self.logsoftmax0(self.input)
        forward_emission = F.embedding(swapped_sentence.reshape(-1), norm_input).reshape(maxlen, batch, self.num_state)
        forward_transition = self.logsoftmax1(self.transition.unsqueeze(0))
        # forward
        forwards = [forward_emission[0]]
        # forwards = [self.tanh(forward_emission[0])]
        for i in range(1, maxlen):
            pre_forward = forwards[i - 1]
            current_forward = self.bmv_log_product(forward_transition, pre_forward)
            forwards.append(current_forward + forward_emission[i])
            # current_forward = torch.matmul(forward_transition, pre_forward.unsqueeze(-1)).squeeze(-1)
            # forwards.append(current_forward * forward_emission[i])
        # shape [batch, max_len, dim]
        hidden_states = torch.stack(forwards)
        length = torch.sum(masks, dim=-1).long()
        forward_scores = hidden_states[length-1, torch.arange(batch)]
        ppl = torch.logsumexp(forward_scores, dim=-1)
        # shape [batch, label]
        # ppl = torch.sum(torch.logsumexp(hidden_states, dim=-1).transpose(0, 1) * masks)
        # score = torch.matmul(expected_count, self.output.unsqueeze(0))
        # nan_detection(score)
        return torch.sum(ppl)

    def get_loss(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        ppl = self.forward(sentences, masks)
        return -1.0 * ppl / masks.size(0)
