import numpy as np
from App.Pre_processing.classifier import Classifier
from App.Pre_processing.gaussian import Gaussian

class FishersLinearDiscriminant(Classifier):
    """
    Fisher's Linear discriminant model
    """

    def __init__(self, w=None, threshold=None):
        self.w = w
        self.threshold = threshold

    def _fit(self, X, t):
        """
        estimate parameter given training dataset

        Parameters
        ----------
        X : (N, D) np.ndarray
            training dataset independent variable
        t : (N,) np.ndarray
            training dataset dependent variable
            binary 0 or 1
        """
        X0 = X[t == 0]
        X1 = X[t == 1]
        m0 = np.average(X0,axis=0)
        m1 = np.average(X1,axis=0)
        s0 = np.zeros((np.size(m0,0), np.size(m0,0)))
        s1 = np.zeros((np.size(m1,0), np.size(m1,0)))
        for i in range(X0.shape[0]):
            k = np.array((X0[i]-m0))[np.newaxis]
            s0 += k.T@k
        for i in range(X1.shape[0]):
            k = np.array((X1[i]-m1))[np.newaxis]
            s1 += k.T@k
        cov_inclass = s0 +s1
        self.w = np.linalg.inv(cov_inclass)@(m1-m0)


        self.w /= np.linalg.norm(self.w).clip(min=1e-10)
        self._threshold(X, t)

    def _threshold(self, X, t):        
        '''
            "minimizing the expected loss"

        '''
        
        X0 = X[t == 0]
        X1 = X[t == 1]
        g0 = Gaussian()
        g0.fit((X0 @ self.w))
        g1 = Gaussian()
        g1.fit((X1 @ self.w))

        # Solving second-order polynomial : "minimizing the expected loss"
        root = np.roots([
            g1.var - g0.var,
            2 * (g0.var * g1.mu - g1.var * g0.mu),
            g1.var * g0.mu ** 2 - g0.var * g1.mu ** 2
            - g1.var * g0.var * np.log(g1.var / g0.var)
        ])

        if g0.mu < root[0] < g1.mu or g1.mu < root[0] < g0.mu:
            self.threshold = root[0]
        else:
            self.threshold = root[1]

    def transform(self, X):
        """
        project data

        Parameters
        ----------
        X : (N, D) np.ndarray
            independent variable

        Returns
        -------
        y : (N,) np.ndarray
            projected data
        """
        return X @ self.w

    def classify(self, X:np.ndarray):
        """
        classify input data

        Parameters
        ----------
        X : (N, D) np.ndarray
            independent variable to be classified

        Returns
        -------
        (N,) np.ndarray
            binary class for each input
        """
        return (X @ self.w > self.threshold).astype(np.int)
