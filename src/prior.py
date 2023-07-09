# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: This file contains the scaling (from the LHS samples) and prior distributions.

import os
import torch
import torch.distributions as td

# our script and functions
import utils.helpers as hp
import setting as st


class ScaleDist:
    def __init__(self, nlhs, factor):
        # store the multiplying factor
        self.factor = factor

        # store the number of LHS points
        self.nlhs = nlhs

        # load LHS samples
        fname = "./lhs/samples_" + str(self.nlhs) + ".csv"

        if os.path.isfile(fname):
            lhs = hp.load_csv("lhs", "samples_" + str(self.nlhs)).iloc[:, 1:]
            self.lhs = torch.tensor(lhs.values, dtype=torch.float)

        else:
            raise Exception(
                "LHS samples not found. Run the LHS script first and store the output in the folder lhs/."
            )

        self.dist = self.distributions(self.factor)

    def distributions(self, factor: float = 5.0) -> dict:
        """Generate the prior distributions for the parameters. The two
        cosmological parameters follow a uniform distribution (see setting
        file).
        The prior for the nuisance parameters follow a multivariate normal distribution.

        Args:
            factor (float, optional): The prior covariance of the nuisance
            parameter is roughly factor times the MLE covariance. Defaults to
            5.0.

        Returns:
            dict: A dictionary of the distributions.
        """

        p0 = td.Uniform(st.prior_om[0], st.prior_om[1], validate_args=False)
        p1 = td.Uniform(st.prior_w0[0], st.prior_w0[1], validate_args=False)

        # the covariance matrix for the nuisance parameters
        # we are using an identity matrix for the covariance of the nuisance parameters
        cov_nui = factor * self.param_cov[2:, 2:]  # factor / factor * torch.eye(4)  #

        # the multivariate normal distribution for the nuisance parameters
        p_nuisance = td.MultivariateNormal(
            self.mle[2:], covariance_matrix=cov_nui, validate_args=False
        )

        return {"p0": p0, "p1": p1, "p_nuisance": p_nuisance}

    def scaling(self):
        # for the uniformly distributed parameters, scaling is trivial
        samples_p0 = self.dist["p0"].icdf(self.lhs[:, 0]).view(self.nlhs, 1)
        samples_p1 = self.dist["p1"].icdf(self.lhs[:, 1]).view(self.nlhs, 1)

        # for the multivariate normal distribution, we use x = m + Lu, where L
        # is the Cholesky decomposition of the covariance matrix
        cov_nui = self.factor * self.param_cov[2:, 2:]
        cho_nui = torch.linalg.cholesky(cov_nui)
        samples_nuisance = (
            self.mle[2:].view(st.n_nui, 1) + cho_nui @ self.lhs[:, 2:].t()
        )

        # concatenate the samples
        samples = torch.cat([samples_p0, samples_p1, samples_nuisance.t()], dim=1)

        return samples
