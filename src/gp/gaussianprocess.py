# Author: Dr. Arrykrishna Mootoovaloo
# Date: 17th January 2022
# Email: arrykrish@gmail.com, a.mootoovaloo17@imperial.ac.uk, arrykrishna.mootoovaloo@physics
# Description: The zero mean Gaussian Process (noise-free implementation, otherwise an extension
# of this is to also supply a noise covariance matrix)

from typing import Union, Tuple
import torch
import torch.autograd
import numpy as np
import src.gp.kernel as kn
import src.gp.transformation as tr


class GaussianProcess(tr.PreWhiten):
    """
    Gaussian Process class

    Args:
        inputs (torch.tensor): the inputs to the GP
        outputs (torch.tensor): the output (target).
        jitter (float): jitter term for numerical stability.
        xtrans (bool, optional): Option to prewhiten the inputs. Defaults to True.
    """

    def __init__(
        self,
        inputs: torch.tensor,
        outputs: torch.tensor,
        jitter: float,
        xtrans: bool = True,
        ytrans: bool = True,
    ):
        # store the relevant informations
        self.xtrans = xtrans
        self.ytrans = ytrans
        if self.ytrans:
            self.ytrain_std = torch.std(outputs)
            self.ytrain_mean = torch.mean(outputs)
            self.ytrain = (outputs - self.ytrain_mean) / self.ytrain_std
            self.ytrain = self.ytrain.view(-1, 1)
        else:
            self.ytrain = outputs
        self.jitter = jitter

        # get the dimensions of the inputs
        self.ndata, self.ndim = inputs.shape

        assert (
            self.ndata > self.ndim
        ), "N < d, please reshape the inputs such that N > d."

        if self.xtrans and self.ndim >= 2:
            tr.PreWhiten.__init__(self, inputs)

            # transform the inputs
            self.xtrain = tr.PreWhiten.x_transformation(self, inputs)

        else:
            self.xtrain = inputs

        # store some variables
        self.d_opt = None
        self.opt_parameters = None
        self.kernel_matrix = None
        self.alpha = None

    def cost(self, parameters: torch.tensor) -> torch.tensor:
        """
        Calculates the negative log-likelihood of the GP.

        Args:
            parameters (torch.tensor): the parameters used to compute the marginal likelihood.

        Returns:
            torch.tensor: the negative log-marginal lielihood.
        """

        # compute the kernel matrix
        kernel = kn.compute(self.xtrain, self.xtrain, parameters)

        # add the jitter term to the kernel matrix
        kernel = kernel + torch.eye(self.xtrain.shape[0]) * self.jitter

        # compute the chi2 and log-determinant of the kernel matrix
        log_marginal = -0.5 * self.ytrain.t() @ kn.solve(
            kernel, self.ytrain
        ) - 0.5 * kn.logdeterminant(kernel)

        return -log_marginal

    def optimisation(
        self,
        parameters: torch.tensor,
        niter: int = 10,
        l_rate: float = 0.01,
        nrestart: int = 5,
    ) -> dict:
        """Optimise for the kernel hyperparameters using Adam in PyTorch.

        Args:
            parameters (torch.tensor): a tensor of the kernel hyperparameters.
            niter (int) : the number of iterations we want to use
            lr (float) : the learning rate
            nrestart (int) : the number of times we want to restart the optimisation

        Returns:
            dict: dictionary consisting of the optimised values of the hyperparameters and the loss.
        """

        dictionary = {}

        for i in range(nrestart):
            # make a copy of the original parameters and perturb it
            params = parameters.clone() + torch.randn(parameters.shape) * 0.1

            # make sure we are differentiating with respect to the parameters
            params.requires_grad = True

            # initialise the optimiser
            optimiser = torch.optim.Adam([params], lr=l_rate)

            loss = self.cost(params)

            # an empty list to store the loss
            record_loss = [loss.item()]

            # run the optimisation
            for _ in range(niter):
                optimiser.zero_grad()

                loss.backward()

                optimiser.step()

                # evaluate the loss
                loss = self.cost(params)

                # record the loss at every step
                record_loss.append(loss.item())

            dictionary[i] = {"parameters": params, "loss": record_loss}

        # get the dictionary for which the loss is the lowest
        self.d_opt = dictionary[
            np.argmin([dictionary[i]["loss"][-1] for i in range(nrestart)])
        ]

        # store the optimised parameters as well
        self.opt_parameters = self.d_opt["parameters"]

        # compute the kernel and store it
        self.kernel_matrix = kn.compute(
            self.xtrain, self.xtrain, self.opt_parameters.data
        )

        # also compute K^-1 y and store it
        self.alpha = kn.solve(self.kernel_matrix, self.ytrain)

        # return the optimised values of the hyperparameters and the loss
        return dictionary

    def gp_prediction(
        self, testpoint: torch.tensor, variance: bool = False
    ) -> Union[Tuple[torch.tensor, torch.tensor], torch.tensor]:
        """Computes the prediction at a given test point.

        Args:
            testpoint (torch.tensor): a tensor of the test point
            variance (bool, optional): if we want to compute the variance as well. Defaults to False.

        Returns:
            Union[Tuple[torch.tensor, torch.tensor], torch.tensor]: The mean and variance or mean only
        """

        testpoint = testpoint.view(-1, 1)
        if self.xtrans and self.ndim >= 2:
            testpoint = tr.PreWhiten.x_transformation(self, testpoint)
        k_star = kn.compute(self.xtrain, testpoint, self.opt_parameters.data)
        mean = k_star.t() @ self.alpha

        if variance:
            k_star_star = kn.compute(testpoint, testpoint, self.opt_parameters.data)
            var = k_star_star - k_star.t() @ kn.solve(self.kernel_matrix, k_star)
            return mean.view(-1), var.view(-1)
        return mean.view(-1)

    def mean_prediction(self, testpoint: torch.tensor) -> torch.tensor:
        """
        Calculates the mean prediction of the GP.

        Args:
            testpoint (torch.tensor): the test point at which we want to compute the prediction.

        Returns:
            torch.tensor: the mean prediction.
        """
        testpoint = testpoint.view(-1, 1)
        if self.xtrans and self.ndim >= 2:
            testpoint = tr.PreWhiten.x_transformation(self, testpoint)

        k_star = kn.compute(self.xtrain, testpoint, self.opt_parameters)
        mean = k_star.t() @ self.alpha
        if self.ytrans:
            mean = mean * self.ytrain_std + self.ytrain_mean
        return mean.view(-1)

    def mean_var_prediction(
        self, testpoint: torch.tensor, variance: bool = False
    ) -> Union[Tuple[torch.tensor, torch.tensor], torch.tensor]:
        """Computes the prediction at a given test point.

        Args:
            testpoint (torch.tensor): a tensor of the test point
            variance (bool, optional): if we want to compute the variance as well. Defaults to False.

        Returns:
            Union[Tuple[torch.tensor, torch.tensor], torch.tensor]: The mean and variance or mean only
        """
        testpoint = testpoint.view(-1, 1)
        if self.xtrans and self.ndim >= 2:
            testpoint = tr.PreWhiten.x_transformation(self, testpoint)
        k_star = kn.compute(self.xtrain, testpoint, self.opt_parameters.data)
        mean = k_star.t() @ self.alpha
        if self.ytrans:
            mean = self.ytrain_std * mean + self.ytrain_mean

        if variance:
            k_star_star = kn.compute(testpoint, testpoint, self.opt_parameters.data)
            var = k_star_star - k_star.t() @ kn.solve(self.kernel_matrix, k_star)
            if self.ytrans:
                var = self.ytrain_std**2 * var
            return mean.view(-1), var.view(-1)
        return mean.view(-1)

    def derivatives(
        self, testpoint: torch.tensor, order: int = 1
    ) -> Union[Tuple[torch.tensor, torch.tensor], torch.tensor]:
        """
        Calculates the first or first/second derivatives of the function.

        Args:
            testpoint (torch.tensor): the test point at which we require the derivatives.
            order (int, optional): the order of the differentiation. Defaults to 1.

        Returns:
            Union[Tuple[torch.tensor, torch.tensor], torch.tensor]: the first or first/second derivatives.
        """

        assert order in [1, 2], "The order of the derivatives must be 1 or 2."
        testpoint.requires_grad = True
        mean = self.mean_prediction(testpoint)
        gradient = torch.autograd.grad(mean, testpoint)
        if order == 1:
            return gradient[0]
        hessian = torch.autograd.functional.hessian(self.mean_prediction, testpoint)
        return gradient[0], hessian
