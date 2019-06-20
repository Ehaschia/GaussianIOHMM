import numpy as np

np.random.seed(0)


def mu_generate(dim, range):
    mat = np.random.rand(dim)
    mat = (mat - 0.5) * 2 * range
    return mat


def cholesky_generate(dim, range):
    mat = np.random.rand(dim, dim)
    mat = (mat - 0.5) * 2 * range
    mat = np.tril(mat)
    return mat


def cholesky_selfmulti(cho_mat):
    return cho_mat.transpose().dot(cho_mat)


def cholesky(mat):
    return np.linalg.cholesky(mat)


def cholesky_det(cho_mat):
    return np.linalg.det(cho_mat)


# just for test, not optimize
def cholesky_inv(cho_mat):
    return np.linalg.inv(cho_mat)


def inv(mat):
    return np.linalg.inv(mat)


def det(mat):
    return np.linalg.det(mat)


def pad(mat, dim, parent=True):
    if parent:
        return np.pad(mat, ((dim, 0), (dim, 0)), 'constant', constant_values=0)
    else:
        return np.pad(mat, ((0, dim), (0, dim)), 'constant', constant_values=0)


# mu2 cho2 is the transition rule
def forward_gaussian_multi(mu1, cho1, mu2, cho2):
    dim = cho1.shape[0]
    # normal format
    lam1 = cholesky_selfmulti(cholesky_inv(cho1))
    lam2 = cholesky_selfmulti(cholesky_inv(cho2))
    # normal format
    eta1 = lam1.dot(mu1)
    eta2 = lam2.dot(mu2)
    # log format
    zeta1 = -0.5 * (dim * np.log(np.pi * 2) + np.log(cholesky_det(cho1) ** 2) + mu1.transpose().dot(lam1.dot(mu1)))
    zeta2 = -0.5 * (
            2 * dim * np.log(np.pi * 2) + np.log(cholesky_det(cho2) ** 2) + mu2.transpose().dot(lam2.dot(mu2)))

    lam1_pand = pad(lam1, dim, parent=False)
    lam_new = lam1_pand + lam2

    eta_new = eta2
    eta_new[:dim] += eta1

    sigma_new = inv(lam_new)
    mu_new = sigma_new.dot(eta_new)

    zeta_new = -0.5 * (
            2 * dim * np.log(np.pi * 2) - np.log(det(lam_new)) + eta_new.transpose().dot(sigma_new.dot(eta_new)))
    # log form
    scale = zeta1 + zeta2 - zeta_new

    return scale, mu_new[dim:], sigma_new[dim:, dim:]


# mu2 cho2 is the transition rule
def backward_gaussian_multi(mu1, cho1, mu2, cho2):
    dim = cho1.shape[0]
    # normal format
    lam1 = cholesky_selfmulti(cholesky_inv(cho1))
    lam2 = cholesky_selfmulti(cholesky_inv(cho2))
    # normal format
    eta1 = lam1.dot(mu1)
    eta2 = lam2.dot(mu2)
    # log format
    zeta1 = -0.5 * (dim * np.log(np.pi * 2) + np.log(cholesky_det(cho1) ** 2) + mu1.transpose().dot(lam1.dot(mu1)))
    zeta2 = -0.5 * (
            2 * dim * np.log(np.pi * 2) + np.log(cholesky_det(cho2) ** 2) + mu2.transpose().dot(lam2.dot(mu2)))

    lam1_pand = pad(lam1, dim, parent=True)
    lam_new = lam1_pand + lam2

    eta_new = eta2
    eta_new[dim:] += eta1

    sigma_new = inv(lam_new)
    mu_new = sigma_new.dot(eta_new)

    zeta_new = -0.5 * (
            2 * dim * np.log(np.pi * 2) - np.log(det(lam_new)) + eta_new.transpose().dot(sigma_new.dot(eta_new)))
    # log form
    scale = zeta1 + zeta2 - zeta_new

    return scale, mu_new[:dim], sigma_new[:dim, :dim]


def gaussian_multi(mu1, cho1, mu2, cho2):
    dim = cho1.shape[0]
    # normal format
    lam1 = cholesky_selfmulti(cholesky_inv(cho1))
    lam2 = cholesky_selfmulti(cholesky_inv(cho2))
    # normal format
    eta1 = lam1.dot(mu1)
    eta2 = lam2.dot(mu2)
    # log format
    zeta1 = -0.5 * (dim * np.log(np.pi * 2) + np.log(cholesky_det(cho1) ** 2) + mu1.transpose().dot(lam1.dot(mu1)))
    zeta2 = -0.5 * (dim * np.log(np.pi * 2) + np.log(cholesky_det(cho2) ** 2) + mu2.transpose().dot(lam2.dot(mu2)))

    lam_new = lam1 + lam2

    eta_new = eta2 + eta1

    sigma_new = inv(lam_new)
    mu_new = sigma_new.dot(eta_new)

    zeta_new = -0.5 * (dim * np.log(np.pi * 2) - np.log(det(lam_new)) + eta_new.transpose().dot(sigma_new.dot(eta_new)))
    # log form
    scale = zeta1 + zeta2 - zeta_new

    return scale, mu_new, sigma_new


# x1 x2 x3
rang = 1.0
dim = 1

x1_cho = np.array([[1.0]])  # cholesky_generate(dim, rang)
x1_mu = np.array([0.0])  # mu_generate(dim, rang)
x2_cho = np.array([[1.0]])  # cholesky_generate(dim, rang)
x2_mu = np.array([1.0])  # mu_generate(dim, rang)

t_rule_cho = np.array([[1.0, 0], [0.5, np.sqrt(3) * 0.5]])  # cholesky_generate(dim * 2, rang)
t_rule_mu = np.array([0.0, 0.0])  # mu_generate(dim * 2, rang)

x1_in_score, x1_in_mu, x1_in_sigma = forward_gaussian_multi(x1_mu, x1_cho, t_rule_mu, t_rule_cho)
x1_in_cho = cholesky(x1_in_sigma)
x2_in_score, x2_in_mu, x2_in_sigma = gaussian_multi(x1_in_mu, x1_in_cho, x2_mu, x2_cho)
# x2_in_cho = cholesky(x2_in_sigma)
in_score = x1_in_score + x2_in_score

x2_out_score, x2_out_mu, x2_out_sigma = backward_gaussian_multi(x2_mu, x2_cho, t_rule_mu, t_rule_cho)
x2_out_cho = cholesky(x2_out_sigma)
x1_out_score, x1_out_mu, x1_out_sigma = gaussian_multi(x2_out_mu, x2_out_cho, x1_mu, x1_cho)
# x1_out_cho = cholesky(x1_out_sigma)
out_score = x2_out_score + x1_out_score

print(in_score)
print(out_score)
