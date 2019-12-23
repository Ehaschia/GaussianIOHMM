import unittest

import math
import copy

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
        # lam = np.array([[4.0 / 3.0, -2.0 / 3.0], [-2.0 / 3.0, 4.0 / 3.0]])
        lam1 = np.linalg.inv(self.sigma1.detach().numpy())
        lam = copy.copy(lam1)
        lam[0][0] += 1.0 / self.sigma0.item()
        golden_sigma = np.linalg.inv(lam)
        self.assertAlmostEqual(golden_sigma[1][1], sigma.item())
        eta0 = 1.0 / self.sigma0.item() * self.mu0.item()
        eta1 = lam1.dot(self.mu1.detach().numpy())
        eta = copy.copy(eta1)
        eta[0] += eta0
        golden_mu = golden_sigma.dot(eta)
        self.assertAlmostEqual(golden_mu[1], mu.item())
        zeta0 = -0.5 * (math.log(2 * math.pi) - np.log(1.0 / self.sigma0.item()) + eta0 * self.sigma0.item() * eta0)
        zeta1 = -0.5 * (2 * math.log(2 * math.pi) - np.log(np.linalg.det(lam1)) + eta1.dot(self.sigma1.detach().numpy().dot(eta1)))
        zeta = -0.5 * (2 * math.log(2 * math.pi) - np.log(np.linalg.det(lam)) + eta.dot(golden_sigma.dot(eta)))
        golden_score = zeta0 + zeta1 - zeta
        self.assertAlmostEqual(golden_score, score.item(), 6)

    def test_one_dim_backward(self):
        score, mu, sigma = gaussian_multi_integral(self.mu1, self.mu0, self.sigma1, self.sigma0, forward=False)
        lam1 = np.linalg.inv(self.sigma1.detach().numpy())
        lam = copy.copy(lam1)
        lam[1][1] += 1.0 / self.sigma0.item()
        golden_sigma = np.linalg.inv(lam)
        self.assertAlmostEqual(golden_sigma[0][0], sigma.item(), 6)
        eta0 = 1.0 / self.sigma0.item() * self.mu0.item()
        eta1 = lam1.dot(self.mu1.detach().numpy())
        eta = copy.copy(eta1)
        eta[1] += eta0
        golden_mu = golden_sigma.dot(eta)
        self.assertAlmostEqual(golden_mu[0], mu.item(), 6)
        zeta0 = -0.5 * (math.log(2 * math.pi) - np.log(1.0 / self.sigma0.item()) + eta0 * self.sigma0.item() * eta0)
        zeta1 = -0.5 * (2 * math.log(2 * math.pi) - np.log(np.linalg.det(lam1)) + eta1.dot(
            self.sigma1.detach().numpy().dot(eta1)))
        zeta = -0.5 * (2 * math.log(2 * math.pi) - np.log(np.linalg.det(lam)) + eta.dot(golden_sigma.dot(eta)))
        golden_score = zeta0 + zeta1 - zeta
        self.assertAlmostEqual(golden_score, score.item(), 6)

if __name__ == '__main__':
    unittest.main()
