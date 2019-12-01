import unittest
from model.LM import MixtureGaussianBatchLanguageModel
from model.basic_operation import *

class TestMixtureGaussianBatchLanguageModel(unittest.TestCase):
    def setUp(self) -> None:
        self.mixture_gaussian_batch_language_model = MixtureGaussianBatchLanguageModel(dim=1, ntokens=2, i_comp_num=1,
                                                                                       t_comp_num=1, o_comp_num=1,
                                                                                       max_comp=1)
        self.input_mu = torch.tensor([[1.0], [0.5]]).float()
        self.input_cho = torch.tensor([[1.0], [1.0]]).float()
        self.trans_mu = torch.tensor([1.0, 0.0]).float()
        self.trans_cho = torch.tensor([[1.0, 1.0], [0.0, 1.0]]).float()
        self.out_mu = torch.tensor([[0.0], [2.0]]).float()
        self.out_cho = torch.tensor([[1.0], [1.0]]).float()
        self.mixture_gaussian_batch_language_model.rewrite_parameter(1, 2, 1, 1, 1, 1, self.input_mu, self.input_cho,
                                                                     self.trans_mu, self.trans_cho, self.out_mu, self.out_cho)

    def test_forward(self):
        sentence = torch.tensor([[0, 1]]).long()
        mask = torch.tensor([[1, 1]]).long()
        real_score = self.mixture_gaussian_batch_language_model.forward(sentence, mask).squeeze().detach().numpy()

        # begin part
        prev_score = torch.zeros(1, requires_grad=False)
        prev_mu = torch.zeros(1, requires_grad=False)
        prev_var = torch.eye(1, requires_grad=False)

        # word1 forward
        word1_part_score, word1_part_mu, word1_part_var = gaussian_multi_integral(self.trans_mu, prev_mu,
                                                                                  atma(self.trans_cho),
                                                                                  prev_var)
        word1_inside_score, word1_inside_mu, word1_inside_var = gaussian_multi(word1_part_mu, self.input_mu[0].unsqueeze(0),
                                                                               word1_part_var, self.input_cho[0].unsqueeze(0) ** 2)
        word1_score = prev_score + word1_part_score + word1_inside_score

        # # word2 forward
        # word2_part_score, word2_part_mu, word2_part_var = gaussian_multi_integral(self.trans_mu, word1_inside_mu,
        #                                                                           atma(self.trans_cho), word1_inside_var)
        # word2_inside_score, word2_inside_mu, word2_inside_var = gaussian_multi(word2_part_mu,
        #                                                                        self.input_pu[1].unsqueeze(0),
        #                                                                        word2_part_var,
        #                                                                        self.input_cho[1].unsqueeze(0) ** 2)
        # word2_score = word1_inside_score + word2_part_score + word2_inside_score

        # word 1 with decode
        # decode 1
        word1_decode_score, word1_decode_mu, word1_decode_var = gaussian_multi(word1_inside_mu, self.out_mu[0],
                                                                               word1_inside_var,
                                                                               self.out_cho[0].unsqueeze(0) ** 2)
        # decode 2
        word2_decode_score, word2_decode_mu, word2_decode_var = gaussian_multi(word1_inside_mu, self.out_mu[1],
                                                                               word1_inside_var,
                                                                               self.out_cho[1].unsqueeze(0) ** 2)
        word1_real_score = word1_decode_score + word1_score
        word2_real_score = word2_decode_score + word1_score
        self.assertAlmostEqual(real_score[0][0], word1_real_score.item())
        self.assertAlmostEqual(real_score[0][1], word2_real_score.item())


if __name__ == '__main__':
    unittest.main()
