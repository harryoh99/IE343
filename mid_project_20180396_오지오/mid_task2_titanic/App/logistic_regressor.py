import numpy as np
import pandas as pd
import math

class LogisticRegressor():


    def __init__(self, w=None):
        self.w = w  
        self.h = 0
    def sigmoid(self, a):
        return 1 / (1 + np.exp(-1 * a))

    def fit(self, X, y, lr, epoch_num):
        self.w = np.random.randn(np.size(X,1))/np.sqrt(np.size(X,1))
        w = self.w
        y = y.reshape(-1,1)
        for _ in range(epoch_num):   
            w_prev = np.copy(w)
            y_hat = self.proba(X)
            grad = X.T @ (y_hat-y)
            w = self.AdaGrad(w, grad, lr)
            if np.allclose(w, w_prev):
                break
            self.w = w
    def AdaGrad(self,w, grad, lr):
        self.h = self.h+ grad*grad
        w = self.w.reshape(-1,1)
        w = w-lr*grad/(np.sqrt(self.h)+ 1e-7)
        return w
    def proba(self, X):
        w = self.w.reshape(-1,1)
        return self.sigmoid(X @ w)

    def predict(self, X):

        return self.proba(X)

    def BCE(self,y, y_hat):
        BCE = -1 * np.mean(y * np.log(y_hat + 1e-7) + (1-y) * np.log(1 - y_hat + 1e-7))
        return BCE

