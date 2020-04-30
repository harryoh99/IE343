import numpy as np
from App.regressor import Regressor



class LinearRegressor(Regressor):
    """
    Linear regression model
    y = X @ w
    t ~ N(t|X @ w, var), we don't need to define var. 
    """

    def _fit(self, X, t):
        self.w = np.linalg.inv(X.T @X) @ X.T @ t
        return self.w


        
       
        
        
    def _predict(self, X, return_std=False):
        y = X @ self.w        

        return y
