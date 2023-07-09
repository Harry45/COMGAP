# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: Generates the training points for building the Gaussian Process.

import torch
import pandas as pd

# our script and functions
import src.prior as sp
import src.compression as sc
import utils.helpers as hp
import setting as st


class simulations(sp.ScaleDist, sc.moped):
    """Perform all the forward simulations.

    Args:
        load_data (bool, optional): Load all the data required for the analysis. Defaults to True.
    """

    def __init__(self, load_data: bool = True, nlhs: int = 10):
        self.nlhs = nlhs

        sc.moped.__init__(self, load_data)
        sp.ScaleDist.__init__(self, self.nlhs, st.FACT)

        # scale the LHS points to the prior
        self.inputs = sp.ScaleDist.scaling(self)

    def precomputations(
        self, expansion_point: torch.tensor = None, save: bool = True
    ) -> None:
        """Computes the MOPED vectors, the theory, gradient and Hessian at the
        expansion point. These are saved in the matrices/ folder.

        Args:
            expansion_point (torch.tensor, optional): Can provide an expansion
            point, otherwise the MLE is used. Defaults to None.
            save (bool, optional): Choose if we want to save the vectors/matrices. Defaults to True.
        """

        if expansion_point is None:
            print(
                "No expansion point provided. Using the Maximum Likelihood Estimator."
            )

            expansion_point = self.mle

        # compute the MOPED vectors
        moped_vectors = sc.moped.vectors(self, expansion_point)

        # compute the model, gradient and hessian at the expansion point
        mu_0, gradient, hessian = sc.moped.theory_expansion(self, expansion_point)

        if save:
            with torch.no_grad():
                hp.store_arrays(moped_vectors.numpy(), "matrices", "moped_vectors")
                hp.store_arrays(mu_0.numpy(), "matrices", "mu_0")
                hp.store_arrays(gradient.numpy(), "matrices", "gradient")
                hp.store_arrays(hessian.numpy(), "matrices", "hessian")
                hp.store_arrays(expansion_point.numpy(), "matrices", "expansion_point")

    def forward(self, save: bool = True) -> None:
        """Run the forward simulations and save the outputs to a csv file in the folder outputs/simulations.

        Args:
            save (bool, optional): [description]. Defaults to True.
        """

        comp_e = torch.zeros((self.nlhs, self.ndim))
        comp_1 = torch.zeros_like(comp_e)
        comp_2 = torch.zeros_like(comp_e)

        for i in range(self.nlhs):
            # compute the exact model, approximate model 1, approximate model 2
            mu_e = sc.moped.theory_only(self, self.inputs[i])
            mu_1 = sc.moped.approximate_theory(self, self.inputs[i], second=False)
            mu_2 = sc.moped.approximate_theory(self, self.inputs[i], second=True)

            # compute the MOPED coefficients
            comp_e[i] = sc.moped.compute_coefficients(self, mu_e)
            comp_1[i] = sc.moped.compute_coefficients(self, mu_1)
            comp_2[i] = sc.moped.compute_coefficients(self, mu_2)

        if save:
            matrix = torch.cat(
                [self.inputs, comp_e, comp_1, comp_2, comp_e - comp_1, comp_e - comp_2],
                dim=1,
            )

            with torch.no_grad():
                data_frame = pd.DataFrame(matrix.numpy(), columns=st.col_names)
                hp.save_pd_csv(
                    data_frame, "simulations", "simulations_" + str(self.nlhs)
                )
