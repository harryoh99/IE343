import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from App.logistic_regressor import LogisticRegressor


def weak_ppl(age):    
    if age<=16:
        return 0
    elif age<=64:
        return 1
    else:
        return 2
    
def age_missing_replace(means, data_frame, Pclass_list):
    for pclass in Pclass_list:
        temp = data_frame['Pclass'] == pclass
        data_frame.loc[temp, 'Age'] = data_frame.loc[temp, 'Age'].fillna(means[pclass]) 

def getData():
    #load and preprocess
    train=pd.read_csv('./Data/titanic_train.csv',index_col=0)
    test=pd.read_csv('./Data/titanic_test.csv',index_col=0)
    train.drop(['Ticket', 'Cabin'],axis = 1, inplace =True)
    test.drop(['Ticket', 'Cabin'],axis = 1, inplace =True)
    
    train['Family'] = train['SibSp']+train['Parch']
    test['Family'] = test['SibSp']+test['Parch'] 
    test.drop(['Parch', 'SibSp'],axis =1, inplace=True)
    train.drop(['Parch', 'SibSp'],axis =1, inplace=True)


    train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    train['Title'] = train['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don','Dona'], 'Others')
    train['Title'] = train['Title'].replace('Ms', 'Miss')
    train['Title'] = train['Title'].replace('Mme', 'Mrs')
    train['Title'] = train['Title'].replace('Mlle', 'Miss')
    test['Title'] = test['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don', 'Dona'], 'Others')
    test['Title'] = test['Title'].replace('Ms', 'Miss')
    test['Title'] = test['Title'].replace('Mme', 'Mrs')
    test['Title'] = test['Title'].replace('Mlle', 'Miss')
    


    test.drop(['Name'],axis=1, inplace=True)
    train.drop(['Name'],axis=1, inplace=True)
    
    Pclass_means = train.groupby('Pclass')['Age'].mean()
    Pclass_list = [1,2,3]

    age_missing_replace(Pclass_means,train,Pclass_list)
    age_missing_replace(Pclass_means,test,Pclass_list)
    train['Age'] = train['Age'].apply(weak_ppl)
    test['Age'] = test['Age'].apply(weak_ppl)

      
    train.Embarked.fillna('S', inplace=True)
    test.Embarked.fillna('S',inplace = True)

    train.Fare.fillna(13.314,inplace=True)
    test.Fare.fillna(13.314,inplace=True)

    

   
    dummies_train= []
    dummies_test = []
    
    cols = ['Pclass', 'Embarked','Sex','Age','Title']
    for col in cols:
        dummies_train.append(pd.get_dummies(train[col]))
        dummies_test.append(pd.get_dummies(test[col]))
    dummies_training = pd.concat(dummies_train, axis=1)
    dummies_testing = pd.concat(dummies_test,axis=1)
    train = pd.concat((train,dummies_training), axis=1)
    test = pd.concat((test,dummies_testing), axis=1)
    train.drop(cols, axis=1, inplace=True)
    test.drop(cols,axis=1, inplace=True)
    
    train_y = train.Survived.copy().to_numpy()
    test_y = test.Survived.copy().to_numpy()
    train_X = train.drop(['Survived'],axis=1).to_numpy()
    test_X = test.drop(['Survived'],axis=1).to_numpy()
    test_X =np.insert(test_X, 0, 1, axis=1)
    train_X=np.insert(train_X, 0, 1, axis=1)

    
    #TODO 
    
    return train_X,train_y,test_X,test_y



def accuracy(true_y,pred_y):
    pred_y[pred_y<0.5]=0
    pred_y[pred_y>=0.5]=1
    accuracy=np.sum(true_y==pred_y)/len(true_y)*100
    print('Accuracy', accuracy,"%" )
    f = open('./result/test_Accuracy.txt', 'w')
    f.write(str(accuracy))
    f.close()
    
if __name__ == "__main__":

    train_X, train_y, test_X, test_y = getData()
    # Model
    model = LogisticRegressor()
    lr= 0.01
    #TODO #learning rate
    epoch= 2000000
    #TODO #epoch number
    # Training
    model.fit(train_X, train_y,lr,epoch)

    # Prediction
    pred_y = model.predict(test_X)

    # Evaluation 
    #CHANGED CODE ONE LINE HERE,
    test_y = test_y.reshape(-1,1)
    accuracy(test_y, pred_y)

    




