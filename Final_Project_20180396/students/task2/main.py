import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from App.logistic_regressor import LogisticRegressor
from App.rbf import *

def getData():
    train=pd.read_csv('./Data/titanic_train.csv',index_col=0)
    test=pd.read_csv('./Data/titanic_test.csv',index_col=0)
    train=train.fillna(0)
    test=test.fillna(0)
    train_X=train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
    train_y=train['Survived'].values
    test_X=test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
    test_y=test['Survived'].values
    train_X=pd.get_dummies(train_X, columns = ['Sex','Embarked'])
    test_X=pd.get_dummies(test_X, columns = ['Sex','Embarked'])
    missing_cols = set(train_X.columns) - set(test_X.columns )
    for c in missing_cols:
        test_X[c] = 0
    test_X = test_X[train_X.columns]
    train_X=train_X.values
    test_X=test_X.values
    return train_X,train_y,test_X,test_y

def accuracy(true_y,pred_y):
    pred_y[pred_y<0.5]=0
    pred_y[pred_y>=0.5]=1
    accuracy=np.sum(true_y==pred_y)/len(true_y)*100
    print('Test Accuracy: ', accuracy,"%" )
    f = open('./result/test_Accuracy.txt', 'w')
    f.write(str(accuracy))
    f.close()

if __name__ == "__main__":
    #Getting the train/test data
    train_X, train_y, test_X, test_y = getData()
    #Constructing the RBF kernel consisting of parameters of all one.
    #No regularization used in this model. 
    kernel = RBF(np.ones(train_X.shape[-1]+1))
    #The input of all inner products between training points via Kernel.
    K = kernel(train_X, train_X)
    # Fit
    #Model for logistic regression
    model = LogisticRegressor()
    #learning rate & epochs initializaiton
    lr=0.005
    epoch_num=5000
    # Training (Learning) the model with the input derived from above
    # Put three inputs : matrix of all inner products between training points using kernel
    #                       ,lr, epoch
    model.fit(K, train_y,lr,epoch_num)
    #For testing, inner products between test and training points, with dimensions boosted up
    #with the kernel function.
    K2 = kernel(test_X,train_X)
    # Predicting the probability of (class being in class 1)each data points
    pred_y = model.predict(K2)
    # Evaluating the accuracy
    accuracy(test_y, pred_y)


