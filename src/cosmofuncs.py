# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: This file contains functions for the cosmology.

from typing import Tuple
import torch
import torch.autograd

# our script and functions
import setting as st


def integration(redshift: torch.tensor, omega_matter: torch.tensor, w_param: torch.tensor) -> torch.tensor:
    """The cosmological function to integrate.

    Args:
        redshift (torch.tensor): the value of the redshift
        omega_matter (torch.tensor): the matter density
        w_param (torch.tensor): the equation of state parameter

    Returns:
        torch.tensor: the value of the cosmological function
    """

    fz = torch.sqrt(omega_matter * (1.0 + redshift) ** 3.0 + (1.0 - omega_matter)
                    * (1.0 + redshift) ** (3.0 * (1.0 + w_param)))

    return 1.0/fz


def forward_cosmo(parameters: torch.tensor, redshift: torch.tensor) -> torch.tensor:
    """Calculate the luminosity distance at a given redshift and cosmological parameters.

    - parameters[0]: omega_matter
    - parameters[1]: w_param

    Args:
        params (torch.tensor): the cosmological parameters (omega_matter, w)
        redshift (torch.tensor): the redshift

    Returns:
        torch.tensor: the luminosity distance
    """

    # the matter density
    omega_matter = parameters[0]

    # the dark energy density
    w_param = parameters[1]

    # multiplying factor
    factor = 10**5 * (1.0 + redshift.item()) * st.speed_light / st.Hubble

    # redshift grid (pytorch does not have an option to do numerical integration)
    # so we use trapezoidal rule from pytorch but we need a redshift grid first
    zgrid = torch.linspace(0.0, redshift.item(), st.nredshift)

    # the function to integrate
    fgrid = integration(zgrid, omega_matter, w_param)

    # the integral
    int_val = 5.0 * torch.log10(factor * torch.trapz(fgrid, zgrid))

    return int_val


def forward_nuisance(parameters: torch.tensor, light_params: torch.tensor) -> torch.tensor:
    """Calculates the theoretical model corresponding to the nuisance parameters.

    - parameters[0]: absolute magnitude
    - parameters[1]: delta magnitude
    - parameters[2]: alpha
    - parameters[3]: beta

    - light_params[0]: log-stellar mass
    - light_params[1]: x1
    - light_params[2]: color

    Args:
        parameters (torch.tensor): A set of nuisance parameters
        light_params (torch.tensor): A set of light curve parameters

    Returns:
        torch.tensor: the value of the model corresponding to the nuisance parameters
    """

    if light_params[0].item() >= 10.0:
        dummy = 1.0
    else:
        dummy = 0.0

    nuisa_model = parameters[0] + dummy * parameters[1] - parameters[2] * light_params[1] + parameters[3] * light_params[2]

    return nuisa_model


def forward(parameters: torch.tensor, redshift: torch.tensor, light_params: torch.tensor) -> torch.tensor:
    """Calculate the full forward model for the problem. This is the sum of the
    cosmological function and the nuisance function.

    - parameters[0]: omega_matter
    - parameters[1]: w_param
    - parameters[2]: absolute magnitude
    - parameters[3]: delta magnitude
    - parameters[4]: alpha
    - parameters[5]: beta

    - light_params[0]: log-stellar mass
    - light_params[1]: x1
    - light_params[2]: color

    Args:
        parameters (torch.tensor): the set of parameters
        redshift (torch.tensor): the redshift
        light_params (torch.tensor): the light curve parameters

    Returns:
        torch.tensor: [description]
    """

    cosmo_params = parameters[0:2]
    nuisa_params = parameters[2:]

    model = forward_nuisance(nuisa_params, light_params) + forward_cosmo(cosmo_params, redshift)

    return model


def first_derivative(parameters: torch.tensor, redshift: torch.tensor, light_params: torch.tensor) -> Tuple[torch.tensor]:
    """Calculate the first derivative of the forward model.

    Args:
        parameters (torch.tensor): the set of parameters
        redshift (torch.tensor): the redshift
        light_params (torch.tensor): the light curve parameters

    Returns:
        Tuple[torch.tensor, torch.tensor]: the forward model value and the first derivative
    """

    parameters.requires_grad = True

    model = forward(parameters, redshift, light_params)

    gradient = torch.autograd.grad(model, parameters)

    return model, gradient[0]


def second_derivative(parameters: torch.tensor, redshift: torch.tensor, light_params: torch.tensor) -> Tuple[torch.tensor]:
    """Calculates the second derivatives of the forward model.

    Args:
        parameters (torch.tensor): the set of parameters
        redshift (torch.tensor): the redshift
        light_params (torch.tensor): the light curve parameters

    Returns:
        Tuple[torch.tensor, torch.tensor, torch.tensor]: the second derivatives of the forward model
    """

    inputs = (parameters, redshift, light_params)

    hessian = torch.autograd.functional.hessian(forward, inputs)

    return hessian[0]
