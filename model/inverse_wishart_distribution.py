from scipy.special import multigammaln
import numpy as np
import torch
from model.basic_operation import atma

_LOG_2 = np.log(2)


class InvWishart():

    @staticmethod
    def logpdf(cho, df, scale):
        """
        Parameters
        ----------
        x : torch.Tensor
            cho of points at which to evaluate the log of the
            probability density function
        df : int
            Degrees of freedom
        scale : ndarray
            Scale matrix
        """

        # Retrieve tr(scale^{-1} x)
        dim = scale.size()[-1]

        if dim > 1:
            cho_inv = torch.inverse(cho)  # works in-place
        else:
            cho_inv = 1. / cho
        log_det_scale = 2 * np.sum(np.log(np.linalg.cholesky(scale).diagonal()))
        log_det_x = 2 * torch.sum(torch.log(torch.abs(cho.diagonal(dim1=1, dim2=2))))
        x_inv = atma(cho_inv)
        tr_scale_x_inv = torch.matmul(scale.view(1, dim, dim), x_inv).diagonal(dim1=1, dim2=2).sum(dim=1)

        # Log PDF
        out = ((0.5 * df * log_det_scale - 0.5 * tr_scale_x_inv) -
               (0.5 * df * dim * _LOG_2 + 0.5 * (df + dim + 1) * log_det_x) -
               multigammaln(0.5 * df, dim))

        return out
