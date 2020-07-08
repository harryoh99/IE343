import numpy as np
import pandas as pd
import math

class LogisticRegressor():
    #INITIALIZATION OF THE MODEL
    def __init__(self,w=None, y=None):
        #First initialize the weight to a none value
        self.w = w
        #self.Y is the value of training y. This will be used in gradients + score functions
        #Initialize to zero.
        self.Y=y
    def fit(self, X, y,lr,epoch_num):
        """
        TRAINING (623,623)    ->(623,624) By including the bias term   
                (N,N)          -> (N,N+1) 
        y dimension is (N,1) so the dimension is (623,1)
        self.w -> (N+1,1)
        model.fit(train_X, train_y,lr,epoch_num) will be called.
        """
        #Here, we save the y value that is used for training
        self.Y =y

        #ADDING THE BIAS TERM 
        #matrix consisted of ones so that we could add it to X
        ones = np.ones((X.shape[0],1))

        #Add the matrix of ones to the X so that we have X E R (N*N+1)
        X = np.concatenate((ones, X),axis=1)
        
        #initialize the weights to one and so that it has the dimension,  (N+1,1)
        self. w= np.ones(np.shape(X[1]))
   
        #Temporarily add one element of one, to use it in the gradient
        #       => Should match the dimension
        #I inserted 1, since, I should make the bias term valid. 
        Y = np.insert(self.Y,0,1)

        #TRAINING PROCESS (TRAINS "epoch_num" times) 
        #epoch_num is a hyperparameter
        for i in range(epoch_num):
            #y_hat = The probability value that we got from the sigmoid_changed function
            y_hat = self.sigmoid_changed(X)

            #Getting the gradient 
            #FYI: Here the gradient is actual gradient * (-1) 
            #The real gradient is -Y*X.T@(y_hat-y) but just to make it pretty, just left it like that
            #Plus, multiplying Y is not essential since, we multiply Y in the changed sigmoid function. 
            #Just did to make it sure.
            gradient = Y*(X.T @(y_hat-y))

            #Perform the gradient descent
            #since the gradient variable is has opposite sign of the real gradient-> add it
            self.w += gradient *lr

            #Printing the training error every 100 iterations.
            if(i%100==0):
                self.accuracy(y,y_hat)



    def predict(self, X):
        #print(self.w.shape)
        """
        Input dimension (N,N)
        Output dimension (N,1)
        """
        #ADDING THE BIAS TERM 
        #matrix consisted of ones so that we could add it to X
        ones = np.ones((X.shape[0],1))
         #Add the matrix of ones to the X so that we have X E R (N*D+1)
        X = np.concatenate((ones, X),axis=1)
        #Call the sigmoid_changed function to get the probability value and return it to the caller.
        return self.sigmoid_changed(X)

    def sigmoid_changed(self,X):
        """
        SIGMOID_CHANGED FUNCTION (MORE OF A COMPUTING SCORE +SIGMOID_CHANGED)
        FIRST using the given X, we compute the score. 
        Then we put this in the sigmoid function 1/1+e^z and return it
        input dimension is (N,N+1)
        z/output dimension is (N,1)
        """
        #Temporarily add one element of one, to use it in the gradient
        #       => Should match the dimension
        #I inserted 1, since, I should make the bias term valid. 
        Y = np.insert(self.Y,0,1)

        #Calculating the score function
        t = self.w*Y
        z = X@t

        #returning the sigmoid function value, which is the probability value.
        #We put the score function value into the sigmoid function.
        return 1/(1+np.exp(z))

#(HEAD) UNUSED IN THIS TASK"""
    def sigmoid(self,X):
        """
        input dimension is (N,d)
        z/output dimension is (N,1)
        """
       # print("IAM",X.shape)
        z = X@ self.w
        return 1/(1+np.exp(z))
#UNUSED IN THIS TASK(TAIL)"""

    '''
        Computing the accuracy by comparing the predicted y and the true y value
        When we get the probability value from the sigmoid_changed function, 
        we check if it is bigger than 0.5-> if yes we put that to class 1
        if not to class 0
    '''
    def accuracy(self,true_y,pred_y):
        #If the predicted probability is smaller than 0.5 -> class 0
        pred_y[pred_y<0.5]=0
        #If the predicted probability is bigger than 0.5 -> class 1
        pred_y[pred_y>=0.5]=1
        #Check how many true y and predicted y are equal among the total datapoints.
        accuracy=np.sum(true_y==pred_y)/len(true_y)*100
        #PRINTING The accuracy
        print('Training Accuracy: ', accuracy,"%" )

