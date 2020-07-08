import numpy as np
from App.Pre_processing.kernel import Kernel


class Euclidean(Kernel):

    def __init__(self, ndim):
        """
        construct Radial basis kernel function

        Parameters
        ----------
        params : (ndim + 1,) ndarray
            parameters of radial basis function

        Attributes
        ----------
        ndim : int
            dimension of expected input data
        """

        self.ndim = 2
        

    def __call__(self, x, y, pairwise=True):
        """
        calculate radial basis function
        k(x, y) = c0 * exp(-0.5 * c1 * (x1 - y1) ** 2 ...)

        Parameters
        ----------
        x : ndarray [..., ndim]
            input of this kernel function
        y : ndarray [..., ndim]
            another input

        Returns
        -------
        output : ndarray
            output of this radial basis function
        """
        assert x.shape[-1] == self.ndim
        assert y.shape[-1] == self.ndim
        
        if pairwise:
            x, y = self._pairwise(x, y)
        d = (x - y) ** 2
        return np.sqrt(np.sum(d, axis=-1))

    def update_parameters(self, updates):
        self.params += updates