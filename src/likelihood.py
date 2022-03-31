# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: This file contains the likelihood function

from typing import Tuple
import torch

# our own scripts and functions
import src.model as sm
import setting as st


class LogLike(sm.AppMag):

    def __init__(self, load_data: bool = True):
        sm.AppMag.__init__(self, load_data)

    def gaussian_neg_log_like(self, parameters: torch.tensor) -> torch.tensor:
        """Computes the negative log-likelihood (cost function or loss or chi2) for the given parameters.

        Args:
            parameters (torch.tensor): the parameters to be used for the likelihood.

        Returns:
            torch.tensor: the negative log-likelihood.
        """

        # compute the theoretical model
        theory = sm.AppMag.theory_only(self, parameters)

        # compute different between the data and the theoretical model
        diff = self.data_vec - theory

        if st.opt_cov:
            data_cov = sm.AppMag.theory_covariance(self, parameters[-2:])
            logdet = torch.logdet(data_cov)

        else:
            data_cov = self.data_cov
            logdet = self.logdet_data_cov

        # compute the negative log likelihood
        nll = 0.5 * diff.t() @ torch.linalg.solve(data_cov, diff) + 0.5 * logdet

        return nll

    def first_der_nll(self, parameters: torch.tensor) -> Tuple[float, torch.tensor]:
        """Calculates the first derivative of the negative log-likelihood (cost function or loss or chi2)

        Args:
            parameters (torch.tensor): the parameters to be used for the gradient computation

        Returns:
            Tuple[float, torch.tensor]: the negative log-likelihood and the gradient
        """

        parameters.requires_grad = True

        nll = self.gaussian_neg_log_like(parameters)

        gradient = torch.autograd.grad(nll, parameters)

        return nll, gradient[0]

    def second_der_nll(self, parameters: torch.tensor) -> torch.tensor:
        """Calculates the second derivative of the negative log-likelihood (cost function or loss or chi2)

        Args:
            parameters (torch.tensor): the parameters to be used for the Hessian computation

        Returns:
            torch.tensor: the hessian matrix of size ndim x ndim
        """

        hessian = torch.autograd.functional.hessian(self.gaussian_neg_log_like, parameters)

        return hessian
