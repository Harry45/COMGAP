# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: In this code, we train the GP model using the training data.

import torch

# our script and functions
from src.gp.gaussianprocess import GaussianProcess
import utils.helpers as hp
import setting as st


class Optimisation(GaussianProcess):

    def __init__(self, simulations: str):

        # load the simulations
        self.csv = hp.load_csv('outputs', simulations)

    def train(self, second: bool = False, jitter: float = 1E-5, xtrans: bool = True, save: bool = False, **kwargs):

        inputs = torch.from_numpy(self.csv[st.cos_and_nui].values)

        if not second:
            outputs = self.csv[st.e_min_f].values

        else:
            outputs = self.csv[st.e_min_s].values

        ndim = inputs.shape[1]

        record_details = list()
        record_gps = list()

        for i in range(ndim):

            # convert to torch tensor
            y = torch.from_numpy(outputs[:, i])

            # create the GP model
            GP_module = GaussianProcess(inputs, y, jitter, xtrans)

            # randomly initialise the hyperparameters
            start = torch.randn((inputs.shape[1] + 1,))

            # train the GP model
            d = GP_module.optimisation(parameters=start, **kwargs)

            # save the results
            record_details.append(d)
            record_gps.append(GP_module)

        if save:

            if not second:
                hp.store_list(record_details, 'outputs', 'gps_first_details_' + str(st.NLHS))
                hp.store_list(record_gps, 'outputs', 'gps_first_modules_' + str(st.NLHS))

            else:
                hp.store_list(record_details, 'outputs', 'gps_second_details_' + str(st.NLHS))
                hp.store_list(record_gps, 'outputs', 'gps_second_modules_' + str(st.NLHS))

        return record_details
