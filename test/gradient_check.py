import numpy as np
import copy

# here dim_n0 = dim_n1


def calculate_zeta(eta, lam):
    dim = eta.size
    return -0.5 * (
            dim * np.log(np.pi * 2) - np.log(np.linalg.det(lam)) + eta.transpose().dot(np.linalg.inv(lam).dot(eta)))


def gaussian_multi(n0, n1):
    mu0, var0 = n0
    mu1, var1 = n1

    # inverse
    lambda0 = np.linalg.inv(var0)
    lambda1 = np.linalg.inv(var1)

    eta0 = lambda0.dot(mu0)
    eta1 = lambda1.dot(mu1)

    zeta0 = calculate_zeta(eta0, lambda0)
    zeta1 = calculate_zeta(eta1, lambda0)

    lambda_new = lambda0 + lambda1

    eta_new = eta1 + eta0

    sigma_new = np.linalg.inv(lambda_new)

    mu_new = sigma_new.dot(eta_new)

    zeta_new = calculate_zeta(eta_new, lambda_new)

    scale = zeta0 + zeta1 - zeta_new

    return scale, mu_new, sigma_new


# score is normal format
def gaussian_score(g, sample):
    mu, var = g
    dim = mu.size
    inv_var = np.linalg.inv(var)

    exp_part = -0.5 * (mu - sample).transpose().dot(inv_var.dot(mu - sample))

    scale_part = np.sqrt(((2 * np.pi) ** dim) * np.linalg.det(var))

    score = (1.0 / scale_part) * np.exp(exp_part)
    return score


def mu_numeric_gradient(n0, n1, sample):
    mu0, var0 = n0
    mu1, var1 = n1

    # check mu0
    delta = 1e-10

    # add delta
    mu_a = copy.copy(mu0)
    mu_a[0] += delta
    # mu_a[1] += deltaa
    scale_a, mu, var = gaussian_multi((mu_a, var0), (mu1, var1))
    score_a = np.exp(scale_a) * gaussian_score((mu, var), sample)

    # minus delta
    mu_b = copy.copy(mu0)
    mu_b[0] -= delta
    # mu_b[1] -= delta
    scale_b, mu, var = gaussian_multi((mu_b, var0), (mu1, var1))
    scale_b = np.exp(scale_b) * gaussian_score((mu, var), sample)

    gradient = (score_a - scale_b) / (2 * delta)
    return gradient


def var_numeric_gradient(n0, n1, sample):
    mu0, var0 = n0
    mu1, var1 = n1

    # check mu0
    delta = 1e-10

    # add delta
    var_a = copy.copy(var0)
    var_a[0][0] += delta
    # mu_a[1] += delta
    scale_a, mu, var = gaussian_multi((mu0, var_a), (mu1, var1))
    score_a = np.exp(scale_a) * gaussian_score((mu, var), sample)

    # minus delta
    var_b = copy.copy(var0)
    var_b[0][0] -= delta
    # mu_b[1] -= delta
    scale_b, mu, var = gaussian_multi((mu0, var_b), (mu1, var1))
    scale_b = np.exp(scale_b) * gaussian_score((mu, var), sample)

    gradient = (score_a - scale_b) / (2 * delta)
    return gradient


# gradient check
def mu_analysis_gradient(n0, n1, sample):
    mu0, var0 = n0
    mu1, var1 = n1

    scale, mu, var = gaussian_multi(n0, n1)
    n_sample = gaussian_score((mu, var), sample)
    gradient = np.exp(scale) * n_sample * np.linalg.inv(var0).dot(sample - mu0)
    return gradient[0]


def var_analysis_gradient(n0, n1, sample):
    mu0, var0 = n0
    mu1, var1 = n1
    dim = mu1.size
    var0_inv = np.linalg.inv(var0)
    scale, mu, var = gaussian_multi(n0, n1)
    n_sample = gaussian_score((mu, var), sample)
    tmp = np.asmatrix(sample - mu0)
    gradient = 0.5 * var0_inv * (tmp.transpose().dot(tmp).dot(var0_inv) - 2.0 * np.eye(dim))
    gradient = np.exp(scale) * n_sample * gradient
    return gradient[0, 0]


# mu0 = np.array([0.0, 0.0])
# var0 = np.array([[1.0, 0.0], [0.0, 1.0]])
# mu1 = np.array([0.0, 0.0])
# var1 = np.array([[1.0, 0.0], [0.0, 1.0]])
#
# sample = np.array([0.0, 0.0])

mu0 = np.array([2.0, 1.0])
var0 = np.array([[3.0, 1.0], [1.0, 3.0]])
mu1 = np.array([0.0, 0.0])
var1 = np.array([[1.0, 0.0], [0.0, 1.0]])

sample = np.array([1.0, 0.0])

g_mu0 = mu_numeric_gradient((mu0, var0), (mu1, var1), sample)
g_mu1 = mu_analysis_gradient((mu0, var0), (mu1, var1), sample)

print(g_mu0)
print(g_mu1)

g_var0 = var_numeric_gradient((mu0, var0), (mu1, var1), sample)
g_var1 = var_analysis_gradient((mu0, var0), (mu1, var1), sample)

print(g_var0)
print(g_var1)


## multiply integral check

def integral_var_numeric_graident(n0, n1, sample):
    mu0, var0 = n0
    mu1, var1 = n1

    # check mu0
    delta = 1e-10

    # add delta
    var_a = copy.copy(var0)
    var_a[0, 0] += delta
    # mu_a[1] += delta
    score_a, mu, var = gaussian_multi((mu0, var_a), (mu1, var1))

    # minus delta
    var_b = copy.copy(var0)
    var_b[0, 0] -= delta
    # mu_b[1] -= delta
    scale_b, mu, var = gaussian_multi((mu0, var_b), (mu1, var1))

    gradient = (np.exp(score_a) - np.exp(scale_b)) / (2 * delta)
    return gradient


def integral_mu_numeric_graident(n0, n1, sample):
    mu0, var0 = n0
    mu1, var1 = n1

    # check mu0
    delta = 1e-10

    # add delta
    mu_a = copy.copy(mu0)
    mu_a[0] += delta
    # mu_a[1] += delta
    score_a, mu, var = gaussian_multi((mu_a, var0), (mu1, var1))

    # minus delta
    mu_b = copy.copy(mu0)
    mu_b[0] -= delta
    # mu_b[1] -= delta
    scale_b, mu, var = gaussian_multi((mu_b, var0), (mu1, var1))

    gradient = (np.exp(score_a) - np.exp(scale_b)) / (2 * delta)
    return gradient


def integral_mu_analysis_grdient(n0, n1, sample):
    mu0, var0 = n0
    mu1, var1 = n1

    scale, mu, var = gaussian_multi(n0, n1)

    var0_inv = np.linalg.inv(var0)

    gradient = np.exp(scale) * var0_inv.dot(mu - mu0)
    # gradient = var0_inv.dot(mu - mu0)
    return gradient[0]


def integral_var_analysis_grdient(n0, n1, sample):
    mu0, var0 = n0
    mu1, var1 = n1
    dim = mu1.size
    var0_inv = np.linalg.inv(var0)
    scale, mu, var = gaussian_multi(n0, n1)
    gradient = 0.5 * np.matmul(var0_inv, (np.matmul(var, var0_inv) - 2.0 * np.eye(dim)))
    gradient = np.exp(scale) * gradient
    return gradient[0, 0]


# mu0 = np.array([0.0, 0.0])
# var0 = np.array([[1.0, 0.0], [0.0, 1.0]])
# mu1 = np.array([0.0, 0.0])
# var1 = np.array([[1.0, 0.0], [0.0, 1.0]])
#
# sample = np.array([0.0, 0.0])

mu0 = np.array([0.0])
var0 = np.array([[1.0]])
mu1 = np.array([0.0])
var1 = np.array([[1.0]])

sample = np.array([0.0])

g2_mu0 = integral_mu_numeric_graident((mu0, var0), (mu1, var1), sample)
g2_mu1 = integral_mu_analysis_grdient((mu0, var0), (mu1, var1), sample)

print('*' * 10)
print(g2_mu0)
print(g2_mu1)

g2_var0 = integral_var_numeric_graident((mu0, var0), (mu1, var1), sample)
g2_var1 = integral_var_analysis_grdient((mu0, var0), (mu1, var1), sample)

print(g2_var0)
print(g2_var1)

print("-" * 10)


# check single gaussian

def var_numeric_check(n, sample):
    delta = 1e-10

    mu, var = n
    var1 = copy.deepcopy(var)
    var1[0, 0] += delta
    score1 = gaussian_score((mu, var1), sample)

    var2 = copy.deepcopy(var)
    var2[0, 0] -= delta
    score2 = gaussian_score((mu, var2), sample)

    gradient = (score1 - score2) / (2 * delta)
    return gradient


def var_analysis_check(n, sample):
    score = gaussian_score(n, sample)

    mu, var = n
    dim = np.size(mu)
    inv_v = np.linalg.inv(var)
    tmp = np.asmatrix(sample - mu)
    tmp = np.matmul(tmp.transpose(), tmp)
    tmp = np.matmul(tmp, inv_v) - 1.0 * np.eye(dim)
    gradient = score * 0.5 * np.matmul(inv_v, tmp)
    return gradient[0, 0]


sample = np.array([0.0])
mu0 = np.array([0.0])
var0 = np.array([[1.0]])

g_var_n = var_numeric_check((mu0, var0), sample)
g_var_a = var_analysis_check((mu0, var0), sample)

print(g_var_n)
print(g_var_a)
