from model.basic_operation import *
import torch.nn as nn
from torch.nn import Parameter
from global_variables import *
from scipy.stats import invwishart
from model.inverse_wishart_distribution import InvWishart


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


# cho init 0 means random, else means init scale
class MixtureGaussianSequenceLabeling(nn.Module):
    def __init__(self, dim: int, ntokens: int, nlabels: int,
                 in_cho_init=0, t_cho_init=0, out_cho_init=0,
                 t_cho_method='random', mu_embedding=None, var_embedding=None,
                 i_comp_num=1, t_comp_num=1, o_comp_num=1, max_comp=1,
                 in_mu_drop=0.0, in_cho_drop=0.0, out_mu_drop=0.0,
                 out_cho_drop=0.0, t_mu_drop=0.0, t_cho_drop=0.0):
        super(MixtureGaussianSequenceLabeling, self).__init__()

        self.dim = dim
        self.ntokens = ntokens
        self.nlabels = nlabels
        self.i_comp_num = i_comp_num
        self.t_comp_num = t_comp_num
        self.o_comp_num = o_comp_num
        self.max_comp = max_comp
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

        self.output_mu = Parameter(torch.empty(self.nlabels, self.o_comp_num * self.dim), requires_grad=True)
        self.output_cho = Parameter(torch.empty(self.nlabels, self.o_comp_num * self.dim),
                                    requires_grad=DECODE_CHO_GRAD)
        self.output_mu_dropout = nn.Dropout(out_mu_drop)
        self.output_cho_dropout = nn.Dropout(out_cho_drop)

        self.criterion = nn.CrossEntropyLoss(reduction='none')

        reset_embedding(mu_embedding, self.input_mu_embedding, self.dim, True, far_init=FAR_EMISSION_MU)

        # init the var
        if in_cho_init != 0:
            var_embedding = torch.empty(self.ntokens, self.i_comp_num*self.dim)
            nn.init.constant_(var_embedding, in_cho_init)
        reset_embedding(var_embedding, self.input_cho_embedding, self.dim, EMISSION_CHO_GRAD, far_init=False)
        self.reset_parameter(trans_cho_method=t_cho_method, output_cho_scale=out_cho_init, t_cho_scale=t_cho_init)

    def reset_parameter(self, trans_cho_method='random', output_cho_scale=0, t_cho_scale=0):
        # transition mu init
        if FAR_TRANSITION_MU:
            nn.init.uniform_(self.transition_mu, a=-1.0, b=1.0)
        else:
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
            transition_var = invwishart.rvs(2*self.dim, np.eye(2*self.dim) / 2*self.dim,
                                            size=self.t_comp_num, random_state=None)
            self.transition_cho.data = torch.from_numpy(np.linalg.cholesky(transition_var)).float()
        else:
            raise ValueError("Error transition init method")
        # output mu init
        if FAR_DECODE_MU:
            nn.init.uniform_(self.output_mu, a=-1.0, b=1.0)
        else:
            nn.init.xavier_normal_(self.output_mu)
        # output var init
        if output_cho_scale == 0:
            nn.init.uniform_(self.output_cho)
        else:
            nn.init.constant_(self.output_cho, output_cho_scale)

    # just used for unit test
    def inject_parameter(self, in_mu, in_cho, tran_mu, tran_cho, out_mu, out_cho):
        self.input_mu_embedding.weight.data = in_mu
        self.input_cho_embedding.weight.data = in_cho
        self.transition_mu.data = tran_mu
        self.transition_cho.data = tran_cho
        self.output_mu.data = out_mu
        self.output_cho.data = out_cho

    #
    # This function is flip backward by sentence length
    # def forward(self, sentences: torch.Tensor, backward_order: torch.Tensor) -> torch.Tensor:
    #     batch, max_len = sentences.size()
    #     swapped_sentences = sentences.transpose(0, 1)
    #
    #     # update cho_embedding to var_embedding
    #     # shape [length, batch, comp, dim]
    #     input_mu_mat = self.input_mu_embedding(swapped_sentences).reshape(max_len, batch, self.i_comp_num, self.dim)
    #     input_var_embedding = (self.input_cho_embedding(swapped_sentences) ** 2).reshape(max_len, batch,
    #                                                                                      self.i_comp_num, self.dim)
    #     input_var_mat = torch.diag_embed(input_var_embedding).float()
    #
    #     output_var = torch.diag_embed(self.output_cho ** 2).float().reshape(self.ntokens, self.o_comp_num,
    #                                                                         self.dim, self.dim)
    #
    #     trans_mu = self.transition_mu.reshape(1, 1, self.t_comp_num, 2*self.dim)
    #     trans_var = atma(self.transition_cho).reshape(1, 1, self.t_comp_num, 2 * self.dim, 2 * self.dim)
    #
    #     # init
    #     # shape [batch, pre_comp, trans_comp, dim, dim]
    #     init_score = torch.zeros(batch, 1, 1)
    #     init_mu = torch.zeros(batch, 1, 1, self.dim, requires_grad=False)
    #     init_var = torch.eye(self.dim, requires_grad=False).repeat(batch, 1, 1).unsqueeze(1).unsqueeze(1)
    #     forward_prev_score = init_score
    #     forward_prev_mu = init_mu
    #     forward_prev_var = init_var
    #     forward_holder_score = [forward_prev_score]
    #     forward_holder_mu = [forward_prev_mu]
    #     forward_holder_var = [forward_prev_var]
    #     # forward
    #     for i in range(max_len):
    #         # update forward score from pred and current inside score
    #         part_score, part_mu, part_var = gaussian_multi_integral(trans_mu, forward_prev_mu, trans_var,
    #                                                                 forward_prev_var, need_zeta=True, forward=True)
    #
    #         forward_score, forward_mu, forward_var = gaussian_multi(
    #             part_mu.reshape(batch, -1, 1, self.dim),
    #             input_mu_mat[i].reshape(batch, 1, self.i_comp_num, self.dim),
    #             part_var.reshape(batch, -1, 1, self.dim, self.dim),
    #             input_var_mat[i].reshape(batch, 1, self.i_comp_num, self.dim, self.dim),
    #             need_zeta=True)
    #
    #         real_forward_score = (part_score + forward_prev_score).reshape(batch, -1, 1) + forward_score
    #
    #         # pruning
    #         forward_prev_score, forward_prev_mu, forward_prev_var = gaussian_top_k_pruning(real_forward_score.reshape(batch, -1),
    #                                                                                        forward_mu.reshape(batch, -1, self.dim),
    #                                                                                        forward_var.reshape(batch, -1, self.dim, self.dim),
    #                                                                                        k=self.max_comp)
    #         forward_holder_score.append(forward_prev_score)
    #         forward_holder_mu.append(forward_mu)
    #         forward_holder_var.append(forward_var)
    #
    #     # shape [len, batch, comp, dim, dim]
    #     forward_scores = torch.stack(forward_holder_score, dim=0)
    #     forward_mus = torch.stack(forward_holder_mu, dim=0)
    #     forward_vars = torch.stack(forward_holder_var, dim=0)
    #
    #     # backward
    #     backward_holder_score = []
    #     backward_holder_mu = []
    #     backward_holder_var = []
    #
    #     backward_prev_score = init_score
    #     backward_prev_mu = init_mu
    #     backward_prev_var = init_var
    #     for i in range(max_len):
    #         part_score, part_mu, part_var = gaussian_multi_integral(trans_mu, backward_prev_mu, trans_var, backward_prev_var,
    #                                                                 forward=False, need_zeta=True)
    #         backward_score, backward_mu, backward_var = gaussian_multi(
    #             part_mu.reshape(batch, -1, 1, self.dim),
    #             torch.gather(input_mu_mat, 0,
    #                          backward_order[:, i].view(batch, 1).repeat(1, self.dim).unsqueeze(0)
    #                          ).reshape(batch, 1, self.i_comp_num, self.dim),
    #             part_var.reshape(batch, -1, 1, self.dim, self.dim),
    #             torch.gather(input_var_mat, 0,
    #                          backward_order[:, i].view(batch, 1, 1).repeat(1, self.dim, self.dim).unsqueeze(
    #                              0)).reshape(batch, 1, self.i_comp_num, self.dim, self.dim),
    #             need_zeta=False)
    #
    #         real_backward_score = (part_score + backward_prev_score).reshape(batch, -1, 1) + backward_score
    #
    #         # pruning
    #         backward_prev_score, backward_prev_mu, backward_prev_var = gaussian_top_k_pruning(
    #             real_backward_score.reshape(batch, -1),
    #             backward_mu.reshape(batch, -1, self.dim),
    #             backward_var.reshape(batch, -1, self.dim, self.dim),
    #             k=self.max_comp)
    #         backward_holder_score.append(backward_prev_score)
    #         backward_holder_mu.append(backward_prev_mu)
    #         backward_holder_var.append(backward_prev_var)
    #
    #     backward_holder_score.append(init_score)
    #     backward_holder_mu.append(init_mu)
    #     backward_holder_var.append(init_var)
    #
    #     backward_scores = torch.stack(backward_holder_score, dim=0)
    #     backward_mus = torch.stack(backward_holder_mu, dim=0)
    #     backward_vars = torch.stack(backward_holder_var, dim=0)
    #
    #     expected_count_score_holder = []
    #     expected_count_mu_holder = []
    #     expected_count_var_holder = []
    #     # here calculate the expected count at position i.
    #     # we need to count previous position forward score i-1
    #     # and next position backward score i+1. Thus we here begin with 1.
    #     for i in range(0, max_len):
    #         prev_forward_score = forward_scores[i]
    #         prev_forward_mu = forward_mus[i]
    #         prev_forward_var = forward_vars[i]
    #
    #         current_input_mu = input_mu_mat[i]
    #         current_input_var = input_var_mat[i]
    #         if i + 1 != max_len:
    #             backward_idx = torch.where(backward_order == i + 1)[1]
    #
    #             next_backward_score = torch.gather(backward_mus, 0, backward_idx)
    #             next_backward_mu = torch.gather(
    #                 backward_mus, 0, backward_idx.view(batch, 1).repeat(1, self.dim).unsqueeze(0)).squeeze()
    #             next_backward_var = torch.gather(backward_vars, 0,
    #                                              backward_idx.view(batch, 1, 1).repeat(1, self.dim, self.dim).unsqueeze(
    #                                                  0)).squeeze()
    #         else:
    #             next_backward_score = backward_scores[i+1]
    #             next_backward_mu = backward_mus[i + 1]
    #             next_backward_var = backward_vars[i + 1]
    #         # shape [batch, prev_comp, t_comp]
    #         part_score, part_mu, part_var = gaussian_multi(prev_forward_mu.reshape(batch, -1, 1, self.dim),
    #                                                        current_input_mu.reshape(batch, 1, self.i_comp_num, self.dim),
    #                                                        prev_forward_var.reshape(batch, -1, 1, self.dim),
    #                                                        current_input_var.reshape(batch, 1, self.i_comp_num, self.dim, self.dim),
    #                                                        need_zeta=True)
    #
    #         part_score = part_score + prev_forward_score.reshape(batch, -1, 1)
    #         expected_count_score, expected_count_mu, expected_count_var = \
    #             gaussian_multi(part_mu.reshape(batch, -1, 1, self.dim),
    #                            next_backward_mu.reshape(batch, 1, -1, self.dim),
    #                            part_var.reshape(batch, -1, 1, self.dim, self.dim),
    #                            next_backward_var.reshape(batch, 1, -1, self.dim, self.dim), need_zeta=True)
    #
    #         real_expected_score = part_score.reshape(batch, -1, 1) + expected_count_score
    #
    #         pruned_score, pruned_mu, pruned_var = gaussian_top_k_pruning(real_expected_score.reshape(batch, -1),
    #                                                                      expected_count_mu.reshape(batch, -1, self.dim),
    #                                                                      expected_count_var.reshape(batch, -1, self.dim, self.dim),
    #                                                                      k=self.max_comp)
    #         expected_count_score_holder.append(pruned_score)
    #         expected_count_mu_holder.append(pruned_mu)
    #         expected_count_var_holder.append(pruned_var)
    #
    #     # shape [batch, length, comp, dim, dim]
    #     expected_count_scores = torch.stack(expected_count_score_holder, dim=1)
    #     expected_count_mus = torch.stack(expected_count_mu_holder, dim=1)
    #     expected_count_vars = torch.stack(expected_count_var_holder, dim=1)
    #
    #     # DEBUG debug the length here.
    #     score, _, _ = gaussian_multi(expected_count_mus.view(batch, max_len-1, 1, -1, 1, self.dim),
    #                                  self.output_mu.view(1, 1, self.ntokens, 1, self.o_comp_num, self.dim),
    #                                  expected_count_vars.view(batch, max_len-1, -1, 1, self.dim, self.dim),
    #                                  output_var.view(1, 1, self.ntokens, 1, self.o_comp_num, self.dim, self.dim),
    #                                  need_zeta=True)
    #
    #     # shape [batch, len-1, vocab_size]
    #     real_score = torch.logsumexp((score + expected_count_scores.view(batch, max_len-1, 1, -1, 1)).view(batch, max_len-1, self.ntokens, -1), dim=-1)
    #     return real_score

    # In this forward we simple revert the sentence
    def forward(self, sentences: torch.Tensor) -> torch.Tensor:
        batch, max_len = sentences.size()
        swapped_sentences = sentences.transpose(0, 1)

        # update cho_embedding to var_embedding
        # shape [length, batch, comp, dim]
        forward_input_mu_mat = self.input_mu_dropout(self.input_mu_embedding(swapped_sentences).
                                                     reshape(max_len, batch, self.i_comp_num, self.dim))
        forward_input_var_embedding = (self.input_cho_dropout(self.input_cho_embedding(swapped_sentences)) ** 2).reshape(max_len, batch, self.i_comp_num, self.dim)
        forward_input_var_mat = torch.diag_embed(forward_input_var_embedding).float()

        # name reuse, ugly. Need to approve
        output_mu = self.output_mu_dropout(self.output_mu)
        output_var = torch.diag_embed(self.output_cho_dropout(self.output_cho).reshape(self.nlabels, self.o_comp_num, self.dim) ** 2).float()

        trans_mu = self.trans_mu_dropout(self.transition_mu).reshape(1, 1, self.t_comp_num, 2 * self.dim)
        trans_var = atma(self.trans_cho_dropout(self.transition_cho)).reshape(1, 1, self.t_comp_num, 2 * self.dim, 2 * self.dim)

        # init
        # shape [batch, pre_comp, trans_comp, dim, dim]
        init_score = torch.zeros(batch, 1, 1).to(sentences.device)
        init_mu = torch.zeros(batch, 1, 1, self.dim, requires_grad=False).to(sentences.device)
        init_var = torch.eye(self.dim,
                             requires_grad=False).repeat(batch, 1, 1).unsqueeze(1).unsqueeze(1).to(sentences.device)

        forward_prev_score = init_score.view(batch, 1, 1)
        forward_prev_mu = init_mu.view(batch, 1, 1, self.dim)
        forward_prev_var = init_var.view(batch, 1, 1, self.dim, self.dim)

        part_score, part_mu, part_var = gaussian_multi_integral(trans_mu, forward_prev_mu, trans_var,
                                                                forward_prev_var, need_zeta=True, forward=True)
        forward_prev_score, forward_prev_mu, forward_prev_var = gaussian_top_k_pruning(part_score.view(batch, -1),
                                                                                       part_mu.view(batch, -1,
                                                                                                    self.dim),
                                                                                       part_var.view(batch, -1,
                                                                                                     self.dim,
                                                                                                     self.dim),
                                                                                       k=self.max_comp)
        forward_holder_score = [forward_prev_score]
        forward_holder_mu = [forward_prev_mu]
        forward_holder_var = [forward_prev_var]

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
                                                                    trans_var.view(1, 1, self.t_comp_num, 2 * self.dim,
                                                                                   2 * self.dim),
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

        # backward
        backward_prev_score = init_score
        backward_prev_mu = init_mu
        backward_prev_var = init_var

        part_score, part_mu, part_var = gaussian_multi_integral(trans_mu, backward_prev_mu, trans_var,
                                                                backward_prev_var, need_zeta=True, forward=False)
        backward_prev_score, backward_prev_mu, backward_prev_var = gaussian_top_k_pruning(part_score.view(batch, -1),
                                                                                          part_mu.view(batch, -1, self.dim),
                                                                                          part_var.view(batch, -1, self.dim, self.dim),
                                                                                          k=self.max_comp)

        backward_holder_score = [backward_prev_score]
        backward_holder_mu = [backward_prev_mu]
        backward_holder_var = [backward_prev_var]

        # revert the sentence
        swapped_reverted_sentences = torch.flip(swapped_sentences, dims=[0])
        backward_input_mu_mat = self.input_mu_embedding(swapped_reverted_sentences).reshape(max_len, batch,
                                                                                            self.i_comp_num, self.dim)
        backward_input_var_embedding = (self.input_cho_embedding(swapped_reverted_sentences) ** 2).reshape(max_len,
                                                                                                           batch,
                                                                                                           self.i_comp_num,
                                                                                                           self.dim)
        backward_input_var_mat = torch.diag_embed(backward_input_var_embedding).float()

        for i in range(max_len):
            backward_score, backward_mu, backward_var = gaussian_multi(
                backward_prev_mu.view(batch, -1, 1, self.dim),
                backward_input_mu_mat[i].view(batch, 1, self.i_comp_num, self.dim),
                backward_prev_var.view(batch, -1, 1, self.dim, self.dim),
                backward_input_var_mat[i].view(batch, 1, self.i_comp_num, self.dim, self.dim),
                need_zeta=True)

            part_score, part_mu, part_var = gaussian_multi_integral(trans_mu.view(1, 1, self.t_comp_num, 2 * self.dim),
                                                                    backward_mu.view(batch, -1, 1, self.dim),
                                                                    trans_var.view(1, 1, self.t_comp_num, 2 * self.dim,
                                                                                   2 * self.dim),
                                                                    backward_var.view(batch, -1, 1, self.dim, self.dim),
                                                                    need_zeta=True, forward=False)

            real_backward_score = (backward_score + backward_prev_score.view(batch, -1, 1)).reshape(batch, -1, 1) + \
                                  part_score

            # pruning
            backward_prev_score, backward_prev_mu, backward_prev_var = gaussian_top_k_pruning(
                real_backward_score.reshape(batch, -1),
                part_mu.reshape(batch, -1, self.dim),
                part_var.reshape(batch, -1, self.dim, self.dim),
                k=self.max_comp)
            backward_holder_score.append(backward_prev_score)
            backward_holder_mu.append(backward_prev_mu)
            backward_holder_var.append(backward_prev_var)

        backward_holder_score = backward_holder_score[::-1]
        backward_holder_mu = backward_holder_mu[::-1]
        backward_holder_var = backward_holder_var[::-1]

        expected_count_score_holder = []
        expected_count_mu_holder = []
        expected_count_var_holder = []
        # here calculate the expected count at position i.
        # we need to count previous position forward score i-1
        # and next position backward score i+1. Thus we here begin with 1.
        for i in range(0, max_len):
            prev_forward_score = forward_holder_score[i]
            prev_forward_mu = forward_holder_mu[i]
            prev_forward_var = forward_holder_var[i]

            current_input_mu = forward_input_mu_mat[i]
            current_input_var = forward_input_var_mat[i]

            next_backward_score = backward_holder_score[i + 1]
            next_backward_mu = backward_holder_mu[i + 1]
            next_backward_var = backward_holder_var[i + 1]
            # shape [batch, prev_comp, t_comp, dim, dim]
            part_score, part_mu, part_var = gaussian_multi(prev_forward_mu.reshape(batch, -1, 1, self.dim),
                                                           current_input_mu.reshape(batch, 1, self.i_comp_num,
                                                                                    self.dim),
                                                           prev_forward_var.reshape(batch, -1, 1, self.dim, self.dim),
                                                           current_input_var.reshape(batch, 1, self.i_comp_num,
                                                                                     self.dim, self.dim),
                                                           need_zeta=True)

            part_score = part_score + prev_forward_score.reshape(batch, -1, 1)
            expected_count_score, expected_count_mu, expected_count_var = \
                gaussian_multi(part_mu.reshape(batch, -1, 1, self.dim),
                               next_backward_mu.reshape(batch, 1, -1, self.dim),
                               part_var.reshape(batch, -1, 1, self.dim, self.dim),
                               next_backward_var.reshape(batch, 1, -1, self.dim, self.dim), need_zeta=True)

            real_expected_score = part_score.view(batch, -1, 1) + expected_count_score + \
                                  next_backward_score.view(batch, 1, -1)

            pruned_score, pruned_mu, pruned_var = gaussian_top_k_pruning(real_expected_score.reshape(batch, -1),
                                                                         expected_count_mu.reshape(batch, -1, self.dim),
                                                                         expected_count_var.reshape(batch, -1, self.dim,
                                                                                                    self.dim),
                                                                         k=self.max_comp)
            expected_count_score_holder.append(pruned_score)
            expected_count_mu_holder.append(pruned_mu)
            expected_count_var_holder.append(pruned_var)

        # shape [batch, length, comp, dim, dim]
        expected_count_scores = torch.stack(expected_count_score_holder, dim=1)
        expected_count_mus = torch.stack(expected_count_mu_holder, dim=1)
        expected_count_vars = torch.stack(expected_count_var_holder, dim=1)

        # DEBUG debug the length here.
        score, _, _ = gaussian_multi(expected_count_mus.view(batch, max_len, 1, -1, 1, self.dim),
                                     output_mu.view(1, 1, self.nlabels, 1, self.o_comp_num, self.dim),
                                     expected_count_vars.view(batch, max_len, 1, -1, 1, self.dim, self.dim),
                                     output_var.view(1, 1, self.nlabels, 1, self.o_comp_num, self.dim, self.dim),
                                     need_zeta=True)

        # shape [batch, len-1, vocab_size]
        real_score = torch.logsumexp(
            (score + expected_count_scores.view(batch, max_len, 1, -1, 1)).view(batch, max_len, self.nlabels, -1),
            dim=-1)
        return real_score

    def get_loss(self, sentences: torch.Tensor, labels: torch.Tensor,
                 masks: torch.Tensor, wishart=False) -> torch.Tensor:
        real_score = self.forward(sentences)
        prob = self.criterion(real_score.view(-1, self.nlabels), labels.view(-1)).reshape_as(labels) * masks
        # the loss format can fine tune. According to zechuan's investigation.
        if wishart:
            # here our gaussian only trans is non-diagonal. Thus this regularization only suit trans.
            trans_cho = self.transition_cho.view(1, 2*self.dim, 2*self.dim)
            reg = torch.sum(InvWishart.logpdf(trans_cho, self.dim, torch.eye(2*self.dim)))
        else:
            reg = 0.0

        return (torch.sum(prob) - reg) / sentences.size(0)

    def get_acc(self, sentences: torch.Tensor, labels: torch.Tensor,
                masks: torch.Tensor) -> Tuple:
        real_score = self.forward(sentences)
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
        # TODO the loss format can fine tune. According to zechuan's investigation.
        return torch.sum(prob) / sentences.size(0)

    def get_acc(self, sentences: torch.Tensor, labels: torch.Tensor,
                masks: torch.Tensor) -> Tuple:
        real_score = self.forward(sentences, masks)
        pred = torch.argmax(real_score, dim=-1).cpu().numpy()
        corr = np.sum(np.equal(pred, labels.cpu().numpy()) * masks.cpu().numpy())
        total = np.sum(masks.cpu().numpy())
        return corr / total, corr
