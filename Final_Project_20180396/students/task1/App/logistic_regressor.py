import numpy as np
import pandas as pd
import math

class LogisticRegressor():
    #INITIALIZATION OF THE MODEL
    def __init__(self,w=None):
        #First initialize the weight to a none value
        self.w = w
    def fit(self, X, y,lr,epoch_num):
        """
        TRAINING
        X dimension is (623,11) 
        X dimension is (N,D) => We will change this to (N,D+1) adding the bias term   
        w dimension is (D+1,1)      
        y dimension is (N,1) so the dimension is (623,1)
        model.fit(train_X, train_y,lr,epoch_num) will be called.
        """
        #ADDING THE BIAS TERM 
        #matrix consisted of ones so that we could add it to X
        ones = np.ones((X.shape[0],1))
        #Add the matrix of ones to the X so that we have X E R (N*D+1)
        X = np.concatenate((ones, X),axis=1)

        
        #Initialization of the weights to zero.
        self. w= np.zeros(np.shape(X[1]))

        #TRAINING PROCESS (TRAINS "epoch_num" times) 
        #epoch_num is a hyperparameter
        for i in range(epoch_num):
            #y_hat = The probability value that we got from the sigmoid function
            y_hat = self.sigmoid(X)

            #Getting the gradient 
            #FYI: Here the gradient is actual gradient * (-1) 
            #The real gradient is -X.T@(y_hat-y) but just to make it pretty, just left it like that
            gradient = X.T @(y_hat-y)
            
            #Perform the gradient descent
            #since the gradient variable is has opposite sign of the real gradient-> add it
            self.w += gradient *lr

            #Printing the training error every 100 iterations.
            if(i%100==0):
                self.accuracy(y,y_hat)



    def predict(self, X):
        """
        Here we predict the probability value for the test data.
        Input dimension (N,d)
        Output dimension (N,1)
        """
        #ADDING THE BIAS TERM 
        #matrix consisted of ones so that we could add it to X
        ones = np.ones((X.shape[0],1))

        #Add the matrix of ones to the X so that we have X E R (N*D+1)
        X = np.concatenate((ones, X),axis=1)

        #Call the sigmoid function to get the probability value and return it to the caller.
        return self.sigmoid(X)

    def sigmoid(self,X):
        """
        SIGMOID FUNCTION (MORE OF A COMPUTING SCORE +SIGMOID)
        FIRST using the given X, we compute the score. 
        Then we put this in the sigmoid function 1/1+e^x and return it
        input dimension is (N,d+1)
        z/output dimension is (N,1)
        """
        #X:(N,d+1), self.w: (d+1,1), z:(N,1)
        #Computing the score value. z = XW (bias term included)
        z = X@ self.w

        #returning the sigmoid function value, which is the probability value.
        #We put the score function value into the sigmoid function.
        return 1/(1+np.exp(z))
    

    def accuracy(self,true_y,pred_y):
        '''
        Computing the accuracy by comparing the predicted y and the true y value
        When we get the probability value from the sigmoid function, 
        we check if it is bigger than 0.5-> if yes we put that to class 1
        if not to class 0
        '''
        #If the predicted probability is smaller than 0.5 -> class 0
        pred_y[pred_y<0.5]=0

        #If the predicted probability is bigger than 0.5 -> class 1
        pred_y[pred_y>=0.5]=1

        #Check how many true y and predicted y are equal among the total datapoints.
        accuracy=np.sum(true_y==pred_y)/len(true_y)*100

        #PRINTING The accuracy
        print('Training Accuracy: ', accuracy,"%" )

