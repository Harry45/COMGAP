# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: This file contains the settings for the program.

import torch

# multiplying factor for the prior
FACT = 5.0

GP_PATH = '../gps'

# path for the data
data_path = '../data/' # './data/'

# path for the MLE solution
mle_path = './optimised'

# path for storing the samples
samples_path = '../samples'

# the Hubble constant
Hubble = 70.0

# Speed of light
speed_light = 299792.458

# number of redshifts to integrate over, [0, z, nredshift]
nredshift = 1000

# nuisance parameters
nuisance = ['abs_mag', 'delta_mag', 'alpha', 'beta']

# number of nuisance parameters
n_nui = len(nuisance)

# cosmological parameters
cosmology = ['omega_matter', 'w']

# number of cosmological parameters
n_cos = len(cosmology)

# both the cosmological and nuisance parameters
cos_and_nui = cosmology + nuisance

# total number of parameters
nparams = len(cos_and_nui)

# use the theoretical covariance matrix in the optimisation procedure
opt_cov = True

# priors for the cosmological parameters (uniform)
prior_om = torch.tensor([1E-3, 0.6])
prior_w0 = torch.tensor([-1.50, 0.0])

# column names for the training data
exact = ['exact_{}'.format(i + 1) for i in range(nparams)]
first_der = ['first_{}'.format(i + 1) for i in range(nparams)]
sec_der = ['second_{}'.format(i + 1) for i in range(nparams)]
e_min_f = ['exact_minus_first_{}'.format(i + 1) for i in range(nparams)]
e_min_s = ['exact_minus_second_{}'.format(i + 1) for i in range(nparams)]

# all the column names in the csv file for the training data
col_names = cos_and_nui + exact + first_der + sec_der + e_min_f + e_min_s

# number of MCMC samples per walker
n_mcmc = 10

# number of walkers
n_walkers = 12
