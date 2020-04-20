from model.basic_operation import *
import torch.nn as nn
from torch.nn import Parameter
from global_variables import *
from scipy.stats import invwishart
from model.inverse_wishart_distribution import InvWishart

# cho init 0 means random, else means init scale
class MixtureGaussianSequenceLabeling(nn.Module):
    def __init__(self, dim: int, ntokens: int, nlabels: int,
                 in_cho_init=0, t_cho_init=0, out_cho_init=0,
                 t_cho_method='random', mu_embedding=None, var_embedding=None,
                 i_comp_num=1, t_comp_num=1, o_comp_num=1, max_comp=1,
                 in_mu_drop=0.0, in_cho_drop=0.0, out_mu_drop=0.0,
                 out_cho_drop=0.0, t_mu_drop=0.0, t_cho_drop=0.0, gaussian_decode=True):
        super(MixtureGaussianSequenceLabeling, self).__init__()

        self.dim = dim
        self.ntokens = ntokens
        self.nlabels = nlabels
        self.i_comp_num = i_comp_num
        self.t_comp_num = t_comp_num
        self.o_comp_num = o_comp_num
        self.max_comp = max_comp
        self.gaussian_decode = gaussian_decode
        # parameter init
        self.input_mu_embedding = nn.Embedding(self.ntokens, self.i_comp_num * self.dim)
        self.input_cho_embedding = nn.Embedding(self.ntokens, self.i_comp_num * self.dim)
        self.input_mu_dropout = nn.Dropout(in_mu_drop)
        self.input_cho_dropout = nn.Dropout(in_cho_drop)

        self.transition_mu = Parameter(torch.empty(self.t_comp_num, 2 * self.dim), requires_grad=True)
        self.transition_cho = Parameter(
            torch.empty(self.t_comp_num, 2 * self.dim, 2 * self.dim), requires_grad=TRANSITION_CHO_GRAD)
        self.trans_mu_dropout = nn.Dropout(t_mu_drop)
        self.trans_cho_dropout = nn.Dropout(t_cho_drop)

        # candidate decode method:
        #   1. gaussian expected likelihood (the gaussian)
        #   2. only use mu as input of a nn layer
        #   3. consine distance
        if gaussian_decode:
            self.output_mu = Parameter(torch.empty(self.nlabels, self.o_comp_num * self.dim), requires_grad=True)
            self.output_cho = Parameter(torch.empty(self.nlabels, self.o_comp_num * self.dim),
                                        requires_grad=DECODE_CHO_GRAD)
            self.output_mu_dropout = nn.Dropout(out_mu_drop)
            self.output_cho_dropout = nn.Dropout(out_cho_drop)
        else:
            # here we only consider one vector per label
            assert self.o_comp_num == 1
            assert self.i_comp_num == 1
            assert self.t_comp_num == 1
            self.decode_layer = nn.Linear(self.dim, self.nlabels)
            self.decode_dropout = nn.Dropout(out_mu_drop)

        self.criterion = nn.CrossEntropyLoss(reduction='none')

        reset_embedding(mu_embedding, self.input_mu_embedding, self.dim, True)

        # init the var
        if in_cho_init != 0:
            var_embedding = torch.empty(self.ntokens, self.i_comp_num * self.dim)
            nn.init.constant_(var_embedding, in_cho_init)
        reset_embedding(var_embedding, self.input_cho_embedding, self.dim, EMISSION_CHO_GRAD)
        self.reset_parameter(trans_cho_method=t_cho_method, output_cho_scale=out_cho_init, t_cho_scale=t_cho_init)

    def reset_parameter(self, trans_cho_method='random', output_cho_scale=0, t_cho_scale=0):
        # transition mu init
        to_init_transition_mu = self.transition_mu.unsqueeze(0)
        nn.init.xavier_normal_(to_init_transition_mu)
        self.transition_mu.data = to_init_transition_mu.squeeze(0)
        # transition var init
        if trans_cho_method == 'random':
            nn.init.uniform_(self.transition_cho)
            weight = self.transition_cho.data - 0.5
            weight = torch.tril(weight)
            self.transition_cho.data = weight + t_cho_scale * torch.eye(2 * self.dim)
        elif trans_cho_method == 'wishart':
            transition_var = invwishart.rvs(2 * self.dim, (4 * self.dim + 1) * np.eye(2 * self.dim),
                                            size=self.t_comp_num, random_state=None)
            self.transition_cho.data = torch.from_numpy(np.linalg.cholesky(transition_var)).float()
        else:
            raise ValueError("Error transition init method")
        # output mu init
        if self.gaussian_decode:
            nn.init.xavier_normal_(self.output_mu)

            # output var init
            if output_cho_scale == 0:
                nn.init.uniform_(self.output_cho, a=0.1, b=1.0)
            else:
                nn.init.constant_(self.output_cho, output_cho_scale)
        else:
            nn.init.xavier_normal_(self.decode_layer.weight)

    # just used for unit test
    def inject_parameter(self, in_mu, in_cho, tran_mu, tran_cho, out_mu, out_cho):
        self.input_mu_embedding.weight.data = in_mu
        self.input_cho_embedding.weight.data = in_cho
        self.transition_mu.data = tran_mu
        self.transition_cho.data = tran_cho
        self.output_mu.data = out_mu
        self.output_cho.data = out_cho

    def cal_forward(self, batch, max_len, swapped_sentences, forward_input_mu_mat, forward_input_var_mat):
        # update cho_embedding to var_embedding
        # shape [length, batch, comp, dim]

        trans_mu = self.trans_mu_dropout(self.transition_mu).reshape(1, 1, self.t_comp_num, 2 * self.dim)
        trans_var = atma(self.trans_cho_dropout(self.transition_cho)).reshape(1, 1, self.t_comp_num, 2 * self.dim, 2 * self.dim)

        # init
        # shape [batch, pre_comp, trans_comp, dim, dim]
        init_mu = torch.zeros(batch, 1, 1, self.dim, requires_grad=False).to(swapped_sentences.device)
        init_var = torch.eye(self.dim, requires_grad=False).repeat(batch, 1, 1).unsqueeze(1).unsqueeze(1).to(swapped_sentences.device)

        part_score, part_mu, part_var = gaussian_multi_integral(trans_mu, init_mu, trans_var,
                                                                init_var, need_zeta=True, forward=True)
        forward_prev_score, forward_prev_mu, forward_prev_var = gaussian_top_k_pruning(part_score.view(batch, -1),
                                                                                       part_mu.view(batch, -1, self.dim),
                                                                                       part_var.view(batch, -1, self.dim, self.dim),
                                                                                       k=self.max_comp)
        forward_holder_score = []
        forward_holder_mu = []
        forward_holder_var = []

        # forward
        for i in range(max_len):
            # update forward score from pred and current inside score
            forward_score, forward_mu, forward_var = gaussian_multi(
                forward_prev_mu.view(batch, -1, 1, self.dim),
                forward_input_mu_mat[i].view(batch, 1, self.i_comp_num, self.dim),
                forward_prev_var.view(batch, -1, 1, self.dim, self.dim),
                forward_input_var_mat[i].view(batch, 1, self.i_comp_num, self.dim, self.dim),
                need_zeta=True)
            part_score, part_mu, part_var = gaussian_multi_integral(trans_mu.view(1, 1, self.t_comp_num, 2 * self.dim),
                                                                    forward_mu.view(batch, -1, 1, self.dim),
                                                                    trans_var.view(1, 1, self.t_comp_num, 2 * self.dim, 2 * self.dim),
                                                                    forward_var.view(batch, -1, 1, self.dim, self.dim),
                                                                    need_zeta=True, forward=True)

            # real_forward_score = (part_score + forward_prev_score).reshape(batch, -1, 1) + forward_score
            real_forward_score = (forward_score + forward_prev_score.view(batch, -1, 1)).view(batch, -1, 1) + part_score

            # pruning
            forward_prev_score, forward_prev_mu, forward_prev_var = gaussian_top_k_pruning(
                real_forward_score.reshape(batch, -1),
                part_mu.reshape(batch, -1, self.dim),
                part_var.reshape(batch, -1, self.dim, self.dim),
                k=self.max_comp)
            forward_holder_score.append(forward_prev_score)
            forward_holder_mu.append(forward_prev_mu)
            forward_holder_var.append(forward_prev_var)
        return torch.stack(forward_holder_score), \
               torch.stack(forward_holder_mu), \
               torch.stack(forward_holder_var)

    def forward(self, sentences: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        batch, max_len = sentences.size()
        swapped_sentences = sentences.transpose(0, 1)

        # get embedding
        forward_input_mu_mat = self.input_mu_dropout(self.input_mu_embedding(swapped_sentences).
                                                     reshape(max_len, batch, self.i_comp_num, self.dim))
        forward_input_var_embedding = (
                self.input_cho_dropout(
                    self.input_cho_embedding(
                        swapped_sentences)) ** 2).reshape(max_len, batch, self.i_comp_num, self.dim)

        forward_input_var_mat = torch.diag_embed(forward_input_var_embedding).float()

        forward_scores, forward_mus, forward_vars = self.cal_forward(batch, max_len, swapped_sentences, forward_input_mu_mat, forward_input_var_mat)
        forward_score = forward_scores[length-1, torch.arange(batch)]
        forward_mu = forward_mus[length-1, torch.arange(batch)]
        forward_var = forward_vars[length-1, torch.arange(batch)]

        if self.gaussian_decode:
            output_mu = self.output_mu_dropout(self.output_mu)
            output_var = torch.diag_embed(
                self.output_cho_dropout(self.output_cho).reshape(self.nlabels, self.o_comp_num, self.dim) ** 2).float()

            score, _, _ = gaussian_multi(forward_mu.view(batch, 1, -1, 1, self.dim),
                                         output_mu.view(1, self.nlabels, 1, self.o_comp_num, self.dim),
                                         forward_var.view(batch, 1, -1, 1, self.dim, self.dim),
                                         output_var.view(1, self.nlabels, 1, self.o_comp_num, self.dim, self.dim),
                                         need_zeta=True)
            real_score = torch.logsumexp((score + forward_score.view(batch, 1, -1, 1)).view(batch, self.nlabels, -1), dim=-1)
        else:
            # thus only one gaussian in this method, expected_count_scores is all same
            real_score = self.decode_dropout(self.decode_layer(forward_mu.view(batch, -1))).view(batch, self.nlabels)
        return real_score

    def get_loss(self, sentences: torch.Tensor, labels: torch.Tensor,
                 length: torch.Tensor, normalize_weight=[0.0, 0.0, 0.0]) -> torch.Tensor:
        real_score = self.forward(sentences, length)
        prob = self.criterion(real_score.view(-1, self.nlabels), labels.view(-1)).reshape_as(labels)
        # the loss format can fine tune. According to zechuan's investigation.
        # here our gaussian only trans is non-diagonal. Thus this regularization only suit trans.
        trans_cho = self.transition_cho.view(-1, 2 * self.dim, 2 * self.dim)
        trans_reg = torch.mean(InvWishart.logpdf(trans_cho, 2 * self.dim, (4 * self.dim + 1) * torch.eye(2 * self.dim)))
        if self.input_cho_embedding.weight.requires_grad:
            in_cho = torch.diag_embed(self.input_cho_embedding.weight.reshape(-1, self.i_comp_num, self.dim)).float()
            in_reg = torch.mean(InvWishart.logpdf(in_cho, self.dim, (2 * self.dim + 1) * torch.eye(self.dim)))
        else:
            in_reg = 0.0
        if self.gaussian_decode:
            if self.output_cho.requires_grad:
                out_cho = torch.diag_embed(self.output_cho.reshape(self.nlabels, self.o_comp_num, self.dim))
                out_reg = torch.mean(InvWishart.logpdf(out_cho, self.dim, (2 * self.dim + 1) * torch.eye(self.dim)))
            else:
                out_reg = 0.0
            reg = normalize_weight[0] * trans_reg + normalize_weight[1] * in_reg + normalize_weight[2] * out_reg
        else:
            reg = normalize_weight[0] * trans_reg + normalize_weight[1] * in_reg

        return (torch.sum(prob) - reg) / sentences.size(0)

    def get_acc(self, sentences: torch.Tensor, labels: torch.Tensor, length: torch.Tensor) -> Tuple:
        real_score = self.forward(sentences, length)
        pred = torch.argmax(real_score, dim=-1).cpu().numpy()
        corr = np.sum(np.equal(pred, labels.cpu().numpy()))
        return corr, pred


# pruning component by threshold.
# ALERT this method only can be used for batch=1
class ThresholdPruningMGSL(nn.Module):
    def __init__(self, dim: int, ntokens: int, nlabels: int,
                 in_cho_init=0, t_cho_init=0, out_cho_init=0,
                 t_cho_method='random', mu_embedding=None, var_embedding=None,
                 i_comp_num=1, t_comp_num=1, o_comp_num=1, threshold=0.1,
                 in_mu_drop=0.0, in_cho_drop=0.0, out_mu_drop=0.0,
                 out_cho_drop=0.0, t_mu_drop=0.0, t_cho_drop=0.0, gaussian_decode=True):
        super(ThresholdPruningMGSL, self).__init__()
        import math
        self.threshold = math.log(threshold)

        self.dim = dim
        self.ntokens = ntokens
        self.nlabels = nlabels
        self.i_comp_num = i_comp_num
        self.t_comp_num = t_comp_num
        self.o_comp_num = o_comp_num

        self.gaussian_decode = gaussian_decode
        # parameter init
        self.input_mu_embedding = nn.Embedding(self.ntokens, self.i_comp_num * self.dim)
        self.input_cho_embedding = nn.Embedding(self.ntokens, self.i_comp_num * self.dim)
        self.input_mu_dropout = nn.Dropout(in_mu_drop)
        self.input_cho_dropout = nn.Dropout(in_cho_drop)

        self.transition_mu = Parameter(torch.empty(self.t_comp_num, 2 * self.dim), requires_grad=True)
        self.transition_cho = Parameter(
            torch.empty(self.t_comp_num, 2 * self.dim, 2 * self.dim), requires_grad=TRANSITION_CHO_GRAD)
        self.trans_mu_dropout = nn.Dropout(t_mu_drop)
        self.trans_cho_dropout = nn.Dropout(t_cho_drop)

        # candidate decode method:
        #   1. gaussian expected likelihood (the gaussian)
        #   2. only use mu as input of a nn layer
        #   3. consine distance
        if gaussian_decode:
            self.output_mu = Parameter(torch.empty(self.nlabels, self.o_comp_num * self.dim), requires_grad=True)
            self.output_cho = Parameter(torch.empty(self.nlabels, self.o_comp_num * self.dim),
                                        requires_grad=DECODE_CHO_GRAD)
            self.output_mu_dropout = nn.Dropout(out_mu_drop)
            self.output_cho_dropout = nn.Dropout(out_cho_drop)
        else:
            # here we only consider one vector per label
            assert self.o_comp_num == 1
            assert self.i_comp_num == 1
            assert self.t_comp_num == 1
            self.decode_layer = nn.Linear(self.dim, self.nlabels)
            self.decode_dropout = nn.Dropout(out_mu_drop)

        self.criterion = nn.CrossEntropyLoss(reduction='none')

        reset_embedding(mu_embedding, self.input_mu_embedding, self.dim, True)

        # init the var
        if in_cho_init != 0:
            var_embedding = torch.empty(self.ntokens, self.i_comp_num * self.dim)
            nn.init.constant_(var_embedding, in_cho_init)
        reset_embedding(var_embedding, self.input_cho_embedding, self.dim, EMISSION_CHO_GRAD)
        self.reset_parameter(trans_cho_method=t_cho_method, output_cho_scale=out_cho_init, t_cho_scale=t_cho_init)

    def reset_parameter(self, trans_cho_method='random', output_cho_scale=0, t_cho_scale=0):
        # transition mu init
        to_init_transition_mu = self.transition_mu.unsqueeze(0)
        nn.init.xavier_normal_(to_init_transition_mu)
        self.transition_mu.data = to_init_transition_mu.squeeze(0)
        # transition var init
        if trans_cho_method == 'random':
            nn.init.uniform_(self.transition_cho)
            weight = self.transition_cho.data - 0.5
            weight = torch.tril(weight)
            self.transition_cho.data = weight + t_cho_scale * torch.eye(2 * self.dim)
        elif trans_cho_method == 'wishart':
            transition_var = invwishart.rvs(2 * self.dim, (4 * self.dim + 1) * np.eye(2 * self.dim),
                                            size=self.t_comp_num, random_state=None)
            self.transition_cho.data = torch.from_numpy(np.linalg.cholesky(transition_var)).float()
        else:
            raise ValueError("Error transition init method")
        # output mu init
        if self.gaussian_decode:
            nn.init.xavier_normal_(self.output_mu)

            # output var init
            if output_cho_scale == 0:
                nn.init.uniform_(self.output_cho, a=0.1, b=1.0)
            else:
                nn.init.constant_(self.output_cho, output_cho_scale)
        else:
            nn.init.xavier_normal_(self.decode_layer.weight)

    def cal_forward(self, max_len, sentence, forward_input_mu_mat, forward_input_var_mat):
        # update cho_embedding to var_embedding
        # shape [length, comp, dim]

        trans_mu = self.trans_mu_dropout(self.transition_mu).reshape(1, self.t_comp_num, 2 * self.dim)
        trans_var = atma(self.trans_cho_dropout(self.transition_cho)).reshape(1, self.t_comp_num, 2 * self.dim, 2 * self.dim)

        # init
        # shape [pre_comp, trans_comp, dim, dim]
        init_mu = torch.zeros(1, 1, self.dim, requires_grad=False).to(sentence.device)
        init_var = torch.eye(self.dim, requires_grad=False).view(1, 1, self.dim, self.dim).to(sentence.device)

        part_score, part_mu, part_var = gaussian_multi_integral(trans_mu, init_mu, trans_var, init_var, need_zeta=True, forward=True)
        forward_prev_score, forward_prev_mu, forward_prev_var = gaussian_max_k_pruning(part_score.view(-1),
                                                                                       part_mu.view(-1, self.dim),
                                                                                       part_var.view(-1, self.dim, self.dim),
                                                                                       k=self.threshold)

        # forward
        for i in range(max_len):
            # update forward score from pred and current inside score
            forward_score, forward_mu, forward_var = gaussian_multi(
                forward_prev_mu.view(-1, 1, self.dim),
                forward_input_mu_mat[i].view(1, self.i_comp_num, self.dim),
                forward_prev_var.view(-1, 1, self.dim, self.dim),
                forward_input_var_mat[i].view(1, self.i_comp_num, self.dim, self.dim),
                need_zeta=True)
            part_score, part_mu, part_var = gaussian_multi_integral(trans_mu.view(1, self.t_comp_num, 2 * self.dim),
                                                                    forward_mu.view(-1, 1, self.dim),
                                                                    trans_var.view(1, self.t_comp_num, 2 * self.dim, 2 * self.dim),
                                                                    forward_var.view(-1, 1, self.dim, self.dim),
                                                                    need_zeta=True, forward=True)

            # real_forward_score = (part_score + forward_prev_score).reshape(batch, -1, 1) + forward_score
            real_forward_score = forward_score + forward_prev_score.view(-1, 1) + part_score

            # pruning
            forward_prev_score, forward_prev_mu, forward_prev_var = gaussian_max_k_pruning(real_forward_score.view(-1),
                                                                                           part_mu.view(-1, self.dim),
                                                                                           part_var.view(-1, self.dim, self.dim),
                                                                                           k=self.threshold)
        return forward_prev_score, forward_prev_mu, forward_prev_var

    def forward(self, sentence: torch.Tensor, slen: int) -> torch.Tensor:
        max_len = sentence.size()[0]
        # get embedding
        forward_input_mu_mat = self.input_mu_dropout(self.input_mu_embedding(sentence).reshape(max_len, self.i_comp_num, self.dim))
        forward_input_var_embedding = (self.input_cho_dropout(self.input_cho_embedding(sentence)) ** 2).reshape(max_len, self.i_comp_num, self.dim)

        forward_input_var_mat = torch.diag_embed(forward_input_var_embedding).float()

        forward_score, forward_mu, forward_var = self.cal_forward(slen, sentence, forward_input_mu_mat, forward_input_var_mat)

        if self.gaussian_decode:
            output_mu = self.output_mu_dropout(self.output_mu)
            output_var = torch.diag_embed(
                self.output_cho_dropout(self.output_cho).reshape(self.nlabels, self.o_comp_num, self.dim) ** 2).float()

            score, _, _ = gaussian_multi(forward_mu.view(1, -1, 1, self.dim),
                                         output_mu.view(self.nlabels, 1, self.o_comp_num, self.dim),
                                         forward_var.view(1, -1, 1, self.dim, self.dim),
                                         output_var.view(self.nlabels, 1, self.o_comp_num, self.dim, self.dim),
                                         need_zeta=True)
            real_score = torch.logsumexp((score + forward_score.view(1, -1, 1)).view(self.nlabels, -1), dim=-1)

        else:
            # thus only one gaussian in this method, expected_count_scores is all same
            real_score = self.decode_dropout(self.decode_layer(forward_mu))
        # shape [batch, len-1, vocab_size]
        return real_score

    def get_loss(self, sentences: torch.Tensor, labels: torch.Tensor, slen: int, normalize_weight=[0.0, 0.0, 0.0]) -> torch.Tensor:
        real_score = self.forward(sentences, slen)
        prob = self.criterion(real_score.view(-1, self.nlabels), labels.view(-1)).reshape_as(labels)
        # the loss format can fine tune. According to zechuan's investigation.
        # here our gaussian only trans is non-diagonal. Thus this regularization only suit trans.
        # TODO here the reg calculate batch times
        trans_cho = self.transition_cho.view(-1, 2 * self.dim, 2 * self.dim)
        trans_reg = torch.mean(InvWishart.logpdf(trans_cho, 2 * self.dim, (4 * self.dim + 1) * torch.eye(2 * self.dim)))
        if self.input_cho_embedding.weight.requires_grad:
            in_cho = torch.diag_embed(self.input_cho_embedding.weight.reshape(-1, self.i_comp_num, self.dim)).float()
            in_reg = torch.mean(InvWishart.logpdf(in_cho, self.dim, (2 * self.dim + 1) * torch.eye(self.dim)))
        else:
            in_reg = 0.0
        if self.gaussian_decode:
            if self.output_cho.requires_grad:
                out_cho = torch.diag_embed(self.output_cho.reshape(self.nlabels, self.o_comp_num, self.dim))
                out_reg = torch.mean(InvWishart.logpdf(out_cho, self.dim, (2 * self.dim + 1) * torch.eye(self.dim)))
            else:
                out_reg = 0.0
            reg = normalize_weight[0] * trans_reg + normalize_weight[1] * in_reg + normalize_weight[2] * out_reg
        else:
            reg = normalize_weight[0] * trans_reg + normalize_weight[1] * in_reg

        return torch.sum(prob) - reg

    def get_acc(self, sentences: torch.Tensor, labels: torch.Tensor, slen: int) -> Tuple:
        real_score = self.forward(sentences, slen)
        pred = torch.argmax(real_score, dim=-1).cpu().numpy()
        corr = 1 if pred == labels.cpu().numpy() else 0
        return corr, pred


class RNNSequenceLabeling(nn.Module):
    def __init__(self, rnn_type, ntokens, nlabels, ninp, nhid, nlayers=1, dropout=0.0, tie_weights=False):
        super(RNNSequenceLabeling, self).__init__()
        self.ntokens = ntokens + 2
        self.nlabels = nlabels
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(self.ntokens, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, batch_first=True, bidirectional=True)
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
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, masks.sum(dim=1), batch_first=True,
                                                       enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_emb)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        decoded = self.decoder(self.drop(output))
        return decoded

    def get_loss(self, sentences: torch.Tensor, labels: torch.Tensor,
                 masks: torch.Tensor) -> torch.Tensor:
        real_score = self.forward(sentences, masks)
        prob = self.criterion(real_score.view(-1, self.nlabels), labels.view(-1)).reshape_as(labels) * masks
        return torch.sum(prob) / sentences.size(0)

    def get_acc(self, sentences: torch.Tensor, labels: torch.Tensor,
                masks: torch.Tensor) -> Tuple:
        real_score = self.forward(sentences, masks)
        pred = torch.argmax(real_score, dim=-1).cpu().numpy()
        corr = np.sum(np.equal(pred, labels.cpu().numpy()) * masks.cpu().numpy())
        return corr, pred
