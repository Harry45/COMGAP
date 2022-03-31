# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: This file contains the optimisation procedure to get the Maximum Likelihood Estimate.

import torch

# our script and functions
import src.likelihood as sl
import utils.helpers as hp


class GradientDescent(sl.LogLike):

    def __init__(self, load_data: bool = True):
        sl.LogLike.__init__(self, load_data)

    def run(self, parameter: torch.tensor, n_iter: iter, eps: float = 1E-5, save: bool = False) -> dict:
        """Calculates the maximum likelihood estimate using the gradient descent method. Breaks when the norm is less than eps.

        Args:
            parameter (torch.tensor): the initial guess for the parameters.
            n_iter (iter): the number of iterations to run the optimisation.
            eps (float, optional): the optimisation stops below this value. Defaults to 1E-5.
            save (bool, optional): whether to save the optimisation results. Defaults to False.

        Returns:
            dict: a dictionary consisting of the optimised parameters and the negative log-likelihood at each iteration.
        """

        record_loss = []

        for i in range(n_iter):

            nll, grad = sl.LogLike.first_der_nll(self, parameter)

            if torch.isnan(torch.tensor([nll.item()])).item():
                print("Nan detected. Aborting Now. Please try a different initial guess.")
                break

            hess = sl.LogLike.second_der_nll(self, parameter)

            print(f"Iteration {i+1:2d}: {nll.item():10.4f}")

            # record important quantities
            record_loss.append(nll.item())

            with torch.no_grad():
                # update the parameters using the gradient descent
                param_dummy = parameter - torch.inverse(hess) @ grad

                # compute the norm of the difference between the new and old parameters
                delta = torch.linalg.norm(param_dummy - parameter)

                # set the new parameters as the old parameters
                parameter = param_dummy

                if delta < eps:
                    print(f"\nConverged after {i+1} iterations.")
                    break

        d = {'loss': record_loss, 'parameters': parameter}

        if save:
            hp.store_arrays(parameter.numpy(), 'optimised', 'mle')
            hp.store_arrays(torch.linalg.inv(hess).numpy(), 'optimised', 'mle_covariance')

        return d
