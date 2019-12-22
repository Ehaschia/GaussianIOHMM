import unittest
from model.sequence_labeling import MixtureGaussianSequenceLabeling
from model import basic_operation
import torch


class TestMixtureGaussianSequenceLabeling(unittest.TestCase):

    def setUp(self) -> None:
        # here I test the expected count is same
        self.init_mu = torch.tensor([0.0])
        self.init_var = torch.tensor([[1.0]])
        self.trans_mu = torch.tensor([0.0, 3.0])
        self.trans_var = torch.tensor([[1.0, 0.4], [0.4, 1.0]])
        self.word1_mu = torch.tensor([3.0])
        self.word1_var = torch.tensor([[0.1]])
        self.word2_mu = torch.tensor([2.0])
        self.word2_var = torch.tensor([[0.2]])
        self.out1_mu = torch.tensor([0.5])
        self.out1_var = torch.tensor([[7.0]])
        self.out2_mu = torch.tensor([-0.5])
        self.out2_var = torch.tensor([[2.0]])

    # used to check the expected count at every position is same
    def test_exp_cnt(self):
        # init model
        # model = MixtureGaussianSequenceLabeling(dim=1, ntokens=2, nlabels=2)

        # forward
        part1_score, part1_mu, part1_var = basic_operation.gaussian_multi_integral(self.trans_mu, self.init_mu,
                                                                                   self.trans_var, self.init_var,
                                                                                   forward=True, need_zeta=True)
        forward1_score, forward1_mu, forward1_var = basic_operation.gaussian_multi(part1_mu, self.word1_mu,
                                                                                   part1_var, self.word1_var,
                                                                                   need_zeta=True)
        part2_score, part2_mu, part2_var = basic_operation.gaussian_multi_integral(self.trans_mu, forward1_mu,
                                                                                   self.trans_var, forward1_var,
                                                                                   forward=True, need_zeta=True)
        forward2_score, forward2_mu, forward2_var = basic_operation.gaussian_multi(part2_mu, self.word2_mu,
                                                                                   part2_var, self.word2_var,
                                                                                   need_zeta=True)

        # backward
        part2b_score, part2b_mu, part2b_var = basic_operation.gaussian_multi_integral(self.trans_mu, self.init_mu,
                                                                                      self.trans_var, self.init_var,
                                                                                      forward=False, need_zeta=True)
        backward2_score, backward2_mu, backward2_var = basic_operation.gaussian_multi(part2b_mu, self.word2_mu,
                                                                                      part2b_var, self.word2_var,
                                                                                      need_zeta=True)
        part1b_score, part1b_mu, part1b_var = basic_operation.gaussian_multi_integral(self.trans_mu, backward2_mu,
                                                                                      self.trans_var, backward2_var,
                                                                                      forward=False, need_zeta=True)
        backward1_score, backward1_mu, backward1_var = basic_operation.gaussian_multi(part1b_mu, self.word1_mu,
                                                                                      part1b_var, self.word1_var,
                                                                                      need_zeta=True)

        # expected count
        exp1_score, exp1_mu, exp1_var = basic_operation.gaussian_multi(forward1_mu, part1b_mu,
                                                                       forward1_var, part1b_var,
                                                                       need_zeta=True)
        # exp1_score_cand, exp1_mu_cand, exp1_var_cand = basic_operation.gaussian_multi(part1_mu, backward1_mu,
        #                                                                               part1_var, backward1_var,
        #                                                                               need_zeta=True)

        exp2_score, exp2_mu, exp2_var = basic_operation.gaussian_multi(forward2_mu, part2b_mu,
                                                                       forward2_var, part2b_var,
                                                                       need_zeta=True)

        self.assertAlmostEqual(exp1_score.item() + forward1_score.item() + part1_score.item() +
                               part1b_score.item() + backward2_score.item() + part2b_score.item(),
                               exp2_score.item() + forward2_score.item() + part2_score.item() +
                               forward1_score.item() + part1_score.item() + part2b_score.item(),
                               5)

    def test_score(self):
        model = MixtureGaussianSequenceLabeling(dim=1, ntokens=2, nlabels=2)
        input_mu = torch.stack([self.word1_mu, self.word2_mu], dim=0)
        input_cho = torch.sqrt(torch.cat([self.word1_var, self.word2_var], dim=0))
        tran_cho = torch.cholesky(self.trans_var, upper=True).unsqueeze(0)
        out_mu = torch.stack([self.out1_mu, self.out2_mu], dim=0)
        out_cho = torch.sqrt(torch.cat([self.out1_var, self.out2_var], dim=0))

        model.inject_parameter(input_mu, input_cho, self.trans_mu.unsqueeze(0), tran_cho, out_mu, out_cho)

        sentence = torch.tensor([[0, 1]])

        model_score = model.forward(sentence).detach().squeeze().numpy()

        # self calculate
        # forward
        part1_score, part1_mu, part1_var = basic_operation.gaussian_multi_integral(self.trans_mu, self.init_mu,
                                                                                   self.trans_var, self.init_var,
                                                                                   forward=True, need_zeta=True)
        forward1_score, forward1_mu, forward1_var = basic_operation.gaussian_multi(part1_mu, self.word1_mu,
                                                                                   part1_var, self.word1_var,
                                                                                   need_zeta=True)
        part2_score, part2_mu, part2_var = basic_operation.gaussian_multi_integral(self.trans_mu, forward1_mu,
                                                                                   self.trans_var, forward1_var,
                                                                                   forward=True, need_zeta=True)
        forward2_score, forward2_mu, forward2_var = basic_operation.gaussian_multi(part2_mu, self.word2_mu,
                                                                                   part2_var, self.word2_var,
                                                                                   need_zeta=True)

        # backward
        part2b_score, part2b_mu, part2b_var = basic_operation.gaussian_multi_integral(self.trans_mu, self.init_mu,
                                                                                      self.trans_var, self.init_var,
                                                                                      forward=False, need_zeta=True)
        backward2_score, backward2_mu, backward2_var = basic_operation.gaussian_multi(part2b_mu, self.word2_mu,
                                                                                      part2b_var, self.word2_var,
                                                                                      need_zeta=True)
        part1b_score, part1b_mu, part1b_var = basic_operation.gaussian_multi_integral(self.trans_mu, backward2_mu,
                                                                                      self.trans_var, backward2_var,
                                                                                      forward=False, need_zeta=True)
        backward1_score, backward1_mu, backward1_var = basic_operation.gaussian_multi(part1b_mu, self.word1_mu,
                                                                                      part1b_var, self.word1_var,
                                                                                      need_zeta=True)

        # expected count
        exp1_score, exp1_mu, exp1_var = basic_operation.gaussian_multi(forward1_mu, part1b_mu,
                                                                       forward1_var, part1b_var,
                                                                       need_zeta=True)

        exp2_score, exp2_mu, exp2_var = basic_operation.gaussian_multi(forward2_mu, part2b_mu,
                                                                       forward2_var, part2b_var,
                                                                       need_zeta=True)

        pos1_label1, _, _ = basic_operation.gaussian_multi(self.out1_mu, exp1_mu,
                                                           self.out1_var, exp1_var, need_zeta=True)
        pos1_label2, _, _ = basic_operation.gaussian_multi(self.out2_mu, exp1_mu,
                                                           self.out2_var, exp1_var, need_zeta=True)

        pos2_label1, _, _ = basic_operation.gaussian_multi(self.out1_mu, exp2_mu,
                                                           self.out1_var, exp2_var, need_zeta=True)
        pos2_label2, _, _ = basic_operation.gaussian_multi(self.out2_mu, exp2_mu,
                                                           self.out2_var, exp2_var, need_zeta=True)


        real_pos1_label1 = pos1_label1.item() + exp1_score.item() + forward1_score.item() + part1_score.item() + part1b_score.item() + backward2_score.item() + part2b_score.item()
        real_pos1_label2 = pos1_label2.item() + exp1_score.item() + forward1_score.item() + part1_score.item() + part1b_score.item() + backward2_score.item() + part2b_score.item()
        real_pos2_label1 = pos2_label1.item() + exp2_score.item() + forward2_score.item() + part2_score.item() + forward1_score.item() + part1_score.item() + part2b_score.item()
        real_pos2_label2 = pos2_label2.item() + exp2_score.item() + forward2_score.item() + part2_score.item() + forward1_score.item() + part1_score.item() + part2b_score.item()

        self.assertAlmostEqual(model_score[0, 0], real_pos1_label1, 5)
        self.assertAlmostEqual(model_score[0, 1], real_pos1_label2, 5)
        self.assertAlmostEqual(model_score[1, 0], real_pos2_label1, 5)
        self.assertAlmostEqual(model_score[1, 1], real_pos2_label2, 5)
        


