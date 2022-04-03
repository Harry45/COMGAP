# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: Loads the trained GP models and compute the MOPED coefficient

import torch

# our scripts and functions
import utils.helpers as hp
from src.compression import moped
import setting as st


class forwardModel(moped):

    def __init__(self, emulator: bool = True, method: str = 'first', variance: bool = False, nlhs: int = 10):

        moped.__init__(self, True)

        method = method.lower()

        assert method in ['exact', 'first', 'second'], 'The method must be either "exact" or "first" or "second".'

        self.method = method
        self.emulator = emulator
        self.variance = variance

        # load the pre-computed gradients, hessian, data at the expansion point
        self.theory_mu_0 = torch.from_numpy(hp.load_arrays('matrices', 'mu_0'))
        self.theory_gradient = torch.from_numpy(hp.load_arrays('matrices', 'gradient'))
        self.theory_hessian = torch.from_numpy(hp.load_arrays('matrices', 'hessian'))
        self.expansion_point = torch.from_numpy(hp.load_arrays('matrices', 'expansion_point'))
        self.moped_vectors = torch.from_numpy(hp.load_arrays('matrices', 'moped_vectors'))

        if self.emulator:
            # load the GPs
            self.gps = hp.read_list(st.GP_PATH, 'gps_' + self.method + '_modules_' + str(nlhs))

        # compress the data
        self.comp_data = moped.compute_coefficients(self, self.data_vec)

    def compression(self, parameters: torch.tensor) -> torch.tensor:
        """Compress the long data vector to just a few numbers depending on the method chosen

        Args:
            parameters (torch.tensor): The input parameters

        Returns:
            torch.tensor: The compressed data vector (will also return the variance on each if we want it)
        """

        if self.method == 'exact':
            theory = self.theory_only(parameters)

        elif self.method == 'first':
            theory = self.approximate_theory(parameters, second=False)

        elif self.method == 'second':
            theory = self.approximate_theory(parameters, second=True)

        coefficients = moped.compute_coefficients(self, theory)

        if self.emulator and self.variance:

            mean = torch.zeros(self.ndim)
            var = torch.zeros(self.ndim)

            for i in range(self.ndim):
                mean[i], var[i] = self.gps[i].prediction(parameters, variance=True)

            coefficients += mean

            return coefficients, var

        if self.emulator and not self.variance:

            # compute the residuals
            mean = [self.gps[i].prediction(parameters, variance=False).item() for i in range(self.ndim)]
            mean = torch.tensor(mean)

            # add the contribution due to the residuals
            coefficients += mean

        return coefficients
