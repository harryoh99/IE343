import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from App.logistic_regressor import LogisticRegressor



def getData():
    train=pd.read_csv('./Data/titanic_train.csv',index_col=0)
    test=pd.read_csv('./Data/titanic_test.csv',index_col=0)
    
    train=train.fillna(0)
    test=test.fillna(0)
    
    
    train_X=train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
    train_y=train['Survived'].values

    test_X=test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
    test_y=test['Survived'].values
    
    train_X=pd.get_dummies(train_X, columns = ['Sex','Embarked'])#문자 value를 encoding
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

    train_X, train_y, test_X, test_y = getData()
        
    # Fit
    model = LogisticRegressor()
    lr=0.00005
    epoch_num=5000
    #print(np.shape(train_X)[1])
   
    
    # Training (Learning)
    model.fit(train_X, train_y,lr,epoch_num)

    # Predicting
    pred_y = model.predict(test_X)
    #print(pred_y.shape)
    # Evaluating
    accuracy(test_y, pred_y)