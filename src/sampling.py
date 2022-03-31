# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: Calculates the log-likelihood of the compressed data.

import torch
import numpy as np
import emcee

# our own scripts and functions
from src.emulation import forwardModel
from src.prior import ScaleDist
import utils.helpers as hp
import setting as st


class MCMC(forwardModel, ScaleDist):

    def __init__(self, emulator: bool = True, method: str = 'first', variance: bool = False):

        forwardModel.__init__(self, emulator, method, variance)

        ScaleDist.__init__(self, st.NLHS, st.FACT)

        self.priors = ScaleDist.distributions(self, st.FACT)

    def loglikelihood(self, parameters: np.ndarray) -> float:
        """Calculates the log-likelihood given a set of parameters.

        Args:
            parameters (np.ndarray): the parameters to be used for the likelihood calculation.

        Returns:
            float: the log-likelihood.
        """

        # calculates the log-prior and exclude point if the parameter is not within the prior
        logp = self.logprior(parameters)

        if not np.isfinite(logp):

            logl = -1E32

        else:

            parameters = torch.tensor(parameters, dtype=torch.float)

            if self.variance:
                mean, variance = forwardModel.compression(self, parameters)
                variance = torch.diag(variance)

            else:
                mean = forwardModel.compression(self, parameters)
                variance = torch.eye(self.ndim)

            dist = torch.distributions.multivariate_normal.MultivariateNormal(mean, variance)

            logl = dist.log_prob(self.comp_data).numpy()

        return logl

    def logprior(self, parameters: np.ndarray) -> float:
        """Calculates the log-prior given a set of parameters.

        Args:
            parameters (np.ndarray): the parameters to be used for the prior calculation.

        Returns:
            float: the log-prior.
        """

        parameters = torch.tensor(parameters, dtype=torch.float)

        logp = self.priors['p0'].log_prob(parameters[0])
        logp += self.priors['p1'].log_prob(parameters[1])
        logp += self.priors['p_nuisance'].log_prob(parameters[2:])
        logp = logp.numpy()

        return logp

    def logposterior(self, parameters: np.ndarray) -> float:
        """Calculates the log-posterior given a set of parameters.

        Args:
            parameters (np.ndarray): the set of parameters to be used in the calculation.

        Returns:
            float: the log-posterior
        """
        # check if all parameters are finite
        bad_param = np.any(np.isnan(parameters)) or np.any(np.isinf(parameters))

        if bad_param:
            return -1E32

        # calculates the log-likelihood
        logl = self.loglikelihood(parameters)

        # calculates the log-prior
        logp = self.logprior(parameters)

        # calculates the log-posterior
        logpost = logl + logp

        # check if log-posterior is finite
        bad_logp = np.isnan(logpost) or np.isinf(logpost)

        if bad_logp:
            return -1E32

        return logpost

    def generate_samples(self, nsamples: int = 10000, nwalkers: int = 12, sampler_name: str = 'test') -> None:
        """Sample the full posterior distribution

        Args:
            nsamples (int): the number of samples to be generated.
            nwalkers (int): the number of walkers to be used.
            sampler_name (str): sampler name has to be specified and the samples will be saved in the samples/ folder
        """

        assert nwalkers >= 2 * self.ndim, "The number of walkers has to be at least twice the number of dimensions"

        # get the mean values
        mean_values = self.mle.numpy()

        # get the step sizes
        eps_values = 0.2 * torch.sqrt(torch.diag(self.param_cov)).numpy()

        # perturb the initial position
        pos = [mean_values + eps_values * np.random.randn(self.ndim) for i in range(nwalkers)]

        # set up the sampler
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.logposterior)
        sampler.run_mcmc(pos, nsamples, progress=True)

        # store the MCMC samples
        hp.store_list(sampler, st.samples_path, sampler_name)
