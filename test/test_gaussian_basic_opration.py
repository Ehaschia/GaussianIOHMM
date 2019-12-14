import unittest

import math

from model.basic_operation import *


class TestAtma(unittest.TestCase):
    def setUp(self) -> None:
        self.mat = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        self.batch_mat = self.mat.unsqueeze(0)
        self.result = [[1.0, 1.0], [1.0, 2.0]]

    def test_mat_multiply(self):
        numpy_multi_result = atma(self.mat).cpu().numpy().tolist()
        self.assertListEqual(self.result, numpy_multi_result)

    def test_batch_mat_multiply(self):
        numpy_multi_result = atma(self.batch_mat).numpy().tolist()
        self.assertListEqual([self.result], numpy_multi_result)


class TestCalculateZeta(unittest.TestCase):

    def setUp(self) -> None:
        self.mu = torch.tensor([0.5])
        self.sigma = torch.tensor([[2.0]])
        self.lam = torch.tensor([[0.5]])
        # eta = lam * mu
        self.eta = torch.tensor([0.25])

    def test_one_dim_zeta_with_mu(self):
        zeta = float(calculate_zeta(self.eta, self.lam, mu=self.mu).numpy())
        self.assertAlmostEqual(zeta, -1.32801212348)

    def test_one_dim_zeta_with_sigma(self):
        zeta = float(calculate_zeta(self.eta, self.lam, sig=self.sigma).numpy())
        self.assertAlmostEqual(zeta, -1.32801212348)


class TestGaussianMulti(unittest.TestCase):
    def setUp(self) -> None:
        self.mu0 = torch.tensor([0.5])
        self.sigma0 = torch.tensor([[2.0]])
        self.mu1 = torch.tensor([1.0 / 3.0])
        self.sigma1 = torch.tensor([[1.0]])

    def test_one_dim_multiply(self):
        score, mu, sigma = gaussian_multi(self.mu0, self.mu1, self.sigma0, self.sigma1)
        self.assertAlmostEqual(sigma.item(), 2.0 / 3.0)
        self.assertAlmostEqual(mu.item(), 7.0 / 18.0)
        self.assertAlmostEqual(score.item(), - (0.5 * math.log(6 * math.pi) + 1.0 / 216.0))


class TestGaussianMultiIntegral(unittest.TestCase):

    def setUp(self) -> None:
        self.mu0 = torch.tensor([0.5])
        self.sigma0 = torch.tensor([[2.0]])
        self.mu1 = torch.tensor([1.0, 2.0])
        self.sigma1 = torch.tensor([[1.0, 0.5], [0.5, 1.0]])

    def test_one_dim_forward(self):
        score, mu, sigma = gaussian_multi_integral(self.mu1, self.mu0, self.sigma1, self.sigma0, forward=True)
        lam = np.array([[4.0 / 3.0, -2.0 / 3.0], [-2.0 / 3.0, 4.0 / 3.0]])
        lam[0][0] += 0.5
        golden_sigma = np.linalg.inv(lam)[1][1]
        self.assertAlmostEqual(golden_sigma, sigma.item())


if __name__ == '__main__':
    unittest.main()
