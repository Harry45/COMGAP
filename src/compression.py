# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: This file contains the MOPED compression algorithm.

import torch

# our scripts and functions
import src.model as sm


class moped(sm.AppMag):
    def __init__(self, load_data: bool = True):
        sm.AppMag.__init__(self, load_data)

        self.moped_vectors = None

    def vectors(self, parameters: torch.tensor) -> torch.tensor:
        """Calculates the MOPED vectors for the given parameters.

        Args:
            parameters (torch.tensor): The parameters to be used for the compression.

        Returns:
            torch.tensor: The MOPED vectors, stacked in a matrix of size (ndata, ndim).
        """

        # compute the forward model and the gradient
        model, grad, hess = sm.AppMag.theory(self, parameters, hessian=True)

        # we do not need the model and the hessian - just need the first derivatives
        del model
        del hess

        # compute C_inverse @ grad
        # sol is of shape 740 x 6
        sol = torch.linalg.solve(self.data_cov, grad)

        # empty matrix to store the MOPED vectors
        moped_vectors = torch.zeros_like(grad)

        for i in range(self.ndim):
            # the first MOPED vector easy to compute
            if i == 0:
                moped_vectors[:, i] = sol[:, 0] / torch.sqrt(grad[:, 0].t() @ sol[:, 0])

            else:
                # create empty matrices to store pre-computations for the MOPED vectors
                dum_num = torch.zeros((self.ndata, i))
                dum_den = torch.zeros((i))

                for j in range(i):
                    dum_num[:, j] = (grad[:, i] @ moped_vectors[:, j]) * moped_vectors[
                        :, j
                    ]
                    dum_den[j] = (grad[:, i].t() @ moped_vectors[:, j]) ** 2

                # the numerator
                moped_num = sol[:, i] - torch.sum(dum_num, dim=1)

                # the denominator term
                moped_den = torch.sqrt(grad[:, i].t() @ sol[:, i] - torch.sum(dum_den))

                # the MOPED vector
                moped_vectors[:, i] = moped_num / moped_den

        # check we are doing everything right
        for i in range(self.ndim):
            for j in range(i + 1):
                prod = moped_vectors[:, i].t() @ self.data_cov @ moped_vectors[:, j]
                print(f"{i} {j} : {prod.item():7.4f}")

        # store the MOPED vectors
        self.moped_vectors = moped_vectors

        return moped_vectors

    def compute_coefficients(self, vector: torch.tensor) -> torch.tensor:
        """Calculates the MOPED coefficients, that is, the compressed data/theory.

        Args:
            vector (torch.tensor): the data/theory vector of size ndata.

        Raises:
            Exception: The MOPED vectors have to be computed first using the vectors() method.

        Returns:
            torch.tensor: The MOPED coefficients, of size ndim.
        """

        if not hasattr(self, "moped_vectors"):
            raise Exception("MOPED vectors have not been computed.")

        moped_coefficients = self.moped_vectors.t() @ vector

        return moped_coefficients
