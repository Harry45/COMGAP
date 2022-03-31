# Author: Dr. Arrykrishna Mootoovaloo
# Date: 17th January 2022
# Email: arrykrish@gmail.com, a.mootoovaloo17@imperial.ac.uk, arrykrishna.mootoovaloo@physics
# Description: all calculations related to the kernel matrix

import torch


def pairwise_distance(x1: torch.tensor, x2: torch.tensor) -> torch.tensor:

    # compute pairwise distances
    # d = torch.cdist(x1, x2, p=2)
    # d = torch.pow(d, 2)
    # cdist does not support gradient (for hessian computation) so we use the following
    # https://nenadmarkus.com/p/all-pairs-euclidean/

    sqrA = torch.sum(torch.pow(x1, 2), 1, keepdim=True).expand(x1.shape[0], x2.shape[0])
    sqrB = torch.sum(torch.pow(x2, 2), 1, keepdim=True).expand(x2.shape[0], x1.shape[0]).t()
    d = sqrA - 2 * torch.mm(x1, x2.t()) + sqrB

    return d


def compute(x1: torch.tensor, x2: torch.tensor, hyper: torch.tensor) -> torch.tensor:
    """Compute the kernel matrix between two sets of points.

    Args:
        x1 (torch.tensor): [N x d] tensor of points.
        x2 (torch.tensor): [M x d] tensor of points.
        hyper (torch.tensor): [d+1] tensor of hyperparameters.

    Returns:
        torch.tensor: a tensor of size [N x M] containing the kernel matrix.
    """

    _, ndim = x1.shape

    # reshape all tensors in the right dimensions
    x2 = x2.view(-1, ndim)

    # for the hyperparameters, we have an amplitude and ndim lengthscales
    hyper = hyper.view(1, ndim + 1)

    # the inputs are scaled by the characteristic lengthscale
    x1 = x1 / torch.exp(hyper[:, 1:])
    x2 = x2 / torch.exp(hyper[:, 1:])

    # compute the pairwise distance
    d = pairwise_distance(x1, x2)

    # compute the kernel
    kernel = torch.exp(hyper[:, 0]) * torch.exp(-0.5 * d)

    return kernel


def solve(kernel: torch.tensor, vector: torch.tensor) -> torch.tensor:
    """Solve the linear system Kx = b.

    Args:
        kernel (torch.tensor): [N x N] kernel matrix.
        vector (torch.tensor): [N x 1] vector.

    Returns:
        torch.tensor: [N x 1] vector.
    """

    solution = torch.linalg.solve(kernel, vector)

    return solution


def logdeterminant(kernel: torch.tensor) -> torch.tensor:
    """Compute the log determinant of the kernel matrix.

    Args:
        kernel (torch.tensor): [N x N] kernel matrix.

    Returns:
        torch.tensor: [1 x 1] vector.
    """

    logdet = torch.logdet(kernel)

    return logdet
