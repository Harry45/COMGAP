# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: Main script for running MCMC chains.

import argparse

# our scripts and functions
from src.sampling import MCMC
from src.trainingpoints import simulations
from src.gp.training import Optimisation
import setting as st


def train_and_gps(nlhs: list):
    """Generates the training points and GP.

    Args:
        nlhs (list): The number of training points.
    """

    # loop over the number of training points

    for n_lhs in nlhs:

        print(f'\nTraining GP for {n_lhs} training points.')
        # generate the training points
        # class_sim = simulations(load_data=True, nlhs=n_lhs)
        # class_sim.precomputations(save=False)
        # class_sim.forward(save=True)

        # train the gps
        class_gp = Optimisation('simulations_' + str(n_lhs), n_lhs)
        class_gp.train(second=False, jitter=1E-5, save=True, niter=1000, l_rate=0.01, nrestart=3)
        class_gp.train(second=True, jitter=1E-5, save=True, niter=1000, l_rate=0.01, nrestart=3)


def mcmc_runs(nlhs: list, nsamples: int = 10):
    """Performs MCMC sampling for different number of training points.

    Args:
        nlhs (list): A list of the number of training points for the emulator.
        nsamples (int, optional): The number of MCMC samples to generate. Defaults to 10.
    """

    for n_lhs in nlhs:

        print(f'\nRunning MCMC for {n_lhs} training points.')
        # using first order expansion
        sampling_f = MCMC(emulator=True, method='first', variance=False, nlhs=n_lhs)
        sampling_f.generate_samples(nsamples, 12, 'emulator_first_' + str(n_lhs) + '_1')
        sampling_f.generate_samples(nsamples, 12, 'emulator_first_' + str(n_lhs) + '_2')

        # using second order expansion
        sampling_s = MCMC(emulator=True, method='second', variance=False, nlhs=n_lhs)
        sampling_s.generate_samples(nsamples, 12, 'emulator_second_' + str(n_lhs) + '_1')
        sampling_s.generate_samples(nsamples, 12, 'emulator_second_' + str(n_lhs) + '_2')


def main(emu: bool, method: str = 'first', fname: str = 'exact', var: bool = False):
    """Generates the MCMC samples and store them to a file.

    Args:
        emu (bool): option to use the emulator.
        method (str): first or second derivatives if we choose to use the emulator. Defaults to 'first'.
        fname (str): name of the MCMC file.
        var (bool, optional): option to use the variance from the GP. Defaults to False.
    """

    # the method in lower case
    method = method.lower()
    sampling = MCMC(emulator=emu, method=method, variance=var)

    # generate the MCMC samples
    sampling.generate_samples(st.n_mcmc, st.n_walkers, fname)


if __name__ == '__main__':
    # PARSER = argparse.ArgumentParser(description='Sample the posterior using the simulator or emulator')

    # PARSER.add_argument('-e', '--emu', type=str, help='Use the emulator', required=True)
    # PARSER.add_argument('-m', '--method', type=str,
    #                     help='Method to use [exact, first, second]', required=False, default='first')
    # PARSER.add_argument('-f', '--fname', type=str, help='The filename for the samples', required=True)
    # PARSER.add_argument('-v', '--var', type=str, help='If we want to use the
    # GP error', required=False, default='false')

    # ARGS = PARSER.parse_args()

    # ARGS.emu = 't' in ARGS.emu.lower()
    # ARGS.var = 't' in ARGS.var.lower()

    # # run the main script
    # main(ARGS.emu, ARGS.method, ARGS.fname, ARGS.var)

    # NUM_LHS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    NUM_LHS = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # train_and_gps(NUM_LHS)
    mcmc_runs(NUM_LHS, 10000)
