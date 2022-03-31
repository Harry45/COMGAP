# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: Main script for running MCMC chains.

import argparse

# our scripts and functions
from src.sampling import MCMC
import setting as st


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
    PARSER = argparse.ArgumentParser(description='Sample the posterior using the simulator or emulator')

    PARSER.add_argument('-e', '--emu', type=str, help='Use the emulator', required=True)
    PARSER.add_argument('-m', '--method', type=str,
                        help='Method to use [exact, first, second]', required=False, default='first')
    PARSER.add_argument('-f', '--fname', type=str, help='The filename for the samples', required=True)
    PARSER.add_argument('-v', '--var', type=str, help='If we want to use the GP error', required=False, default='false')

    ARGS = PARSER.parse_args()

    ARGS.emu = 't' in ARGS.emu.lower()
    ARGS.var = 't' in ARGS.var.lower()

    # run the main script
    main(ARGS.emu, ARGS.method, ARGS.fname, ARGS.var)
