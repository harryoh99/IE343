import numpy as np
import pandas as pd
import math

class LogisticRegressor():
   def __init__(self, w=None):
      self.w = w
      self.num_class=3
   def fit(self, X, y,lr,epoch_num):
      #TODO : should train 
      num_input = np.shape(X)[1]
      self.w = np.random.randn(num_input,self.num_class)/np.sqrt(num_input * self.num_class)
      #self.b = np.random.randn(self.num_class)/np.sqrt(self.num_class)
      w = self.w
      for _ in range(epoch_num):
         w_prev = np.copy(w)
         score = X @ w
         yhat = self.softmax(score)
         #prediction with shape of (104,)
         loss,grad  = self.cross_entropy(X,yhat,y)
         w = self.SGD(w, grad, lr)
         self.w =w
         if np.allclose(w, w_prev):
                break
         
      


      
   def get_class(self,z):
      return np.argmax(z,axis= 1)
   def predict(self, X):
      #TODO: should predict
      return X@self.w
   def softmax(self,z):
      #Regularization
      z -= np.max(z)
      z_exp = np.exp(z)
      for i in range(z_exp.shape[0]):
         z_exp[i] = z_exp[i]/np.sum(z_exp[i])
      return z_exp
   
   def SGD(self,w, grad, lr):
      w -= lr*grad
      return w
   def one_hot_encoding(self,y):
      y_one_hot_encoded = (np.arange(np.max(y) + 1) == y[:, None]).astype(float)
      return y_one_hot_encoded
   
   def cross_entropy(self,x,yhat, y):
      #TODO: loss function
      size = x.shape[0]
      y_one_hot_encoded= self.one_hot_encoding(y)
      #both now shape is (104,)
      loss= - np.sum(np.log(yhat) * y_one_hot_encoded, axis=1)

      grad = x.T@(y_one_hot_encoded - yhat) / (-1*size)
      return loss, grad

    

