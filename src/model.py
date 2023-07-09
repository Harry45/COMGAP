# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: This file contains the forward model.

import os
import pandas as pd
import numpy as np
import torch
from typing import Union, Tuple
import glob
from astropy.io import fits

# our own scripts and functions
import setting as st
import src.cosmofuncs as cf
import utils.helpers as hp


class AppMag(object):
    """Calculates the forward model.

    Args:
        load_data (bool, optional): Load all the required data. Defaults to True.
    """

    def __init__(self, load_data: bool = True):
        self.ndim = len(st.cos_and_nui)

        if load_data:
            self.load_data()
            self.load_expansion()

    def load_data(self):
        """Load the data from files."""

        # load the data
        light_curve = pd.read_csv(st.data_path + "jla_lcparams.txt", sep=" ", header=0)

        # number of data points
        self.ndata = light_curve.shape[0]

        # we need to convert the data to torch tensors
        self.data_vec = torch.tensor(light_curve["mb"].values, dtype=torch.float)

        # the redshift values
        self.redshift = torch.tensor(light_curve["zcmb"].values, dtype=torch.float)

        # log stellar mass values
        self.log_stellar_mass = torch.tensor(light_curve["3rdvar"].values)

        # x1 values
        self.x1 = torch.tensor(light_curve["x1"].values, dtype=torch.float)

        # color values
        self.color = torch.tensor(light_curve["color"].values, dtype=torch.float)

        # this is a large matrix of size (740x3) x (740x3), which needs to be processed to get the covariance matrix
        self.cov_eta = torch.tensor(
            sum([fits.open(mat)[0].data for mat in glob.glob(st.data_path + "C*.fits")])
        )

        self.sigma_mu = torch.tensor(np.loadtxt(st.data_path + "sigma_mu.txt"))

    def load_expansion(self):
        """Load the maximum likelihood estimate"""

        fname = st.mle_path + "/" + "mle.npz"

        if os.path.isfile(fname):
            # the Maximum Likelihood Estimator
            mle = hp.load_arrays(st.mle_path, "mle")

            # the covariance matrix of the parameters in the optimisation procedure
            param_cov = hp.load_arrays(st.mle_path, "mle_covariance")

            # all the quantities as tensors
            self.mle = torch.tensor(mle, dtype=torch.float)

            # the parameter covariance matrix
            self.param_cov = torch.tensor(param_cov, dtype=torch.float)

            # compute the data covariance matrix at the expansion point
            self.data_cov = self.theory_covariance(self.mle[-2:])

            # compute the log-determinant and store it
            self.logdet_data_cov = torch.logdet(self.data_cov)

        else:
            raise Exception(
                "The maximum likelihood estimator is not available. Please run the optimisation procedure."
            )

    def theory_covariance(self, parameters: torch.tensor) -> torch.tensor:
        """Computes the theoretical covariance matrix according to the explanation in the Betoule et al. paper.

        Args:
            parameters (torch.tensor): The set of parameters consisting of alpha and beta

        Returns:
            torch.tensor: The covariance matrix of the model.
        """

        alpha = parameters[0]
        beta = parameters[1]

        # the covariance matrix
        cov = torch.zeros((self.ndata, self.ndata))

        for i, coef1 in enumerate([1.0, alpha, -beta]):
            for j, coef2 in enumerate([1.0, alpha, -beta]):
                cov += (coef1 * coef2) * self.cov_eta[i::3, j::3]

        # Add diagonal term from Eq. 13
        sigma_pecvel = (5 * 150 / 3e5) / (np.log(10.0) * self.sigma_mu[:, 2])
        cov += torch.eye(self.ndata) * (
            self.sigma_mu[:, 0] ** 2 + self.sigma_mu[:, 1] ** 2 + sigma_pecvel**2
        )

        return cov

    def theory_only(self, parameters: torch.tensor) -> torch.tensor:
        """Compute the theoretical model only (not the derivatives.)

        Args:
            parameters (torch.tensor): The set of parameters.

        Returns:
            torch.tensor: a tensor of size ndata containing the model predictions.
        """
        predicted = torch.zeros(self.ndata, dtype=torch.float)

        for i in range(self.ndata):
            # get the redshift in tensor format
            redshift = self.redshift[i].view(1)

            # the light curve parameters
            light_params = torch.tensor(
                [self.log_stellar_mass[i], self.x1[i], self.color[i]]
            )

            predicted[i] = cf.forward(parameters, redshift, light_params)

        return predicted

    def theory(
        self, parameters: torch.tensor, hessian: bool = False
    ) -> Union[torch.tensor, torch.tensor, torch.tensor]:
        """Calculate the full forward model at every redshift. The gradient is also returned by default.
        We can also calculate the second derivatives.

        Args:
            parameters (dict): the set of parameters
            hessian (bool, optional): the second derivatives. Defaults to False.

        Returns:
            Union[Tuple[torch.tensor, torch.tensor], Tuple[torch.tensor]]: model, gradient OR model, gradient, hessian
        """

        predicted = torch.zeros(self.ndata, dtype=torch.float)
        gradients = torch.zeros((self.ndata, self.ndim), dtype=torch.float)

        for i in range(self.ndata):
            # get the redshift in tensor format
            redshift = self.redshift[i].view(1)

            # the light curve parameters
            light_params = torch.tensor(
                [self.log_stellar_mass[i], self.x1[i], self.color[i]]
            )

            # get the predictions and the first derivatives
            predicted[i], gradients[i] = cf.first_derivative(
                parameters, redshift, light_params
            )

        if hessian:
            hess = torch.zeros((self.ndata, self.ndim, self.ndim), dtype=torch.float)

            for i in range(self.ndata):
                # get the redshift in tensor format
                redshift = self.redshift[i].view(1)

                # the light curve parameters
                light_params = torch.tensor(
                    [self.log_stellar_mass[i], self.x1[i], self.color[i]]
                )

                # get the predictions and the first derivatives
                sec_der = cf.second_derivative(parameters, redshift, light_params)

                # we need the derivatives in the form of a tensor only for the parameters, hence 0
                hess[i] = sec_der[0]

            return predicted, gradients, hess

        return predicted, gradients

    def theory_expansion(
        self, parameters: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Calculates the theory at the expansion point. Note that this is
        typically done once and the following will be stored:
        1) the expansion point
        2) the model at the expansion point
        3) the gradient at the expansion point
        4) the Hessian at the expansion point

        Args:
            parameters (dict): the set of parameters

        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor]: the model, gradient, Hessian at the expansion point
        """

        # compute the model, gradient and hessian at the expansion point
        mu_0, gradient, hessian = self.theory(parameters, hessian=True)

        # store all the tensors
        self.expansion_point = parameters
        self.theory_mu_0 = mu_0
        self.theory_gradient = gradient
        self.theory_hessian = hessian

        return mu_0, gradient, hessian

    def approximate_theory(
        self, parameters: torch.tensor, second: bool = False
    ) -> torch.tensor:
        """Compute the approximate theory at every redshift, given a set of parameters other than the MLE.

        Args:
            parameters (dict): the set of parameters
            second (bool, optional): [description]. Choose if we want to add the
            term due to the second derivatives. Defaults to False.


        Returns:
            torch.tensor: the approximate theory
        """

        if not hasattr(self, "theory_mu_0"):
            raise Exception(
                "The model, gradient and Hessian have not been computed at the expansion point."
            )

        with torch.no_grad():
            param_diff = parameters - self.expansion_point

            approximate_model = self.theory_mu_0 + self.theory_gradient @ param_diff

            if second:
                # we need to make sure the difference in parameter is a 2D tensor
                param_diff = param_diff.unsqueeze(0)

                # the first product
                prod = param_diff @ self.theory_hessian

                prod = prod.view(self.ndata, self.ndim) @ param_diff.t()

                # add the term due to the second derivatives to the approximate model
                approximate_model += 0.5 * prod.view(self.ndata)

        return approximate_model
