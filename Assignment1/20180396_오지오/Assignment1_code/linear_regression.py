import numpy as np
import matplotlib.pyplot as plt

from App.polynomial import PolynomialFeatures
from App.data_import import create_data_np, filename_generation
from App.linear_regressor import LinearRegressor
import os

def mse(y, y_hat):
    mse_value = np.mean(np.square(y-y_hat))
    return mse_value

def average_mse(mse_list):
    average_mse = np.mean(mse_list)
    std_mse = np.sqrt(np.mean(np.square(mse_list - average_mse)))

    return (average_mse, std_mse)

def plot(x_train, y_train, x_test, y_test, x_linspace, y_hat, filename):
    plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
    plt.scatter(x_test, y_test, facecolor="none", edgecolor="r", s=50, label="test data")
    plt.plot(x_linspace, y_hat, 'r-', label="prediction")
    plt.legend()

    os.chdir("./Results/Result3")
    plt.savefig("./" + filename + ".png")
    os.chdir("..")
    os.chdir("..")
    
    plt.show()
    plt.clf()

def write_file(feature_dimension, average_mse, mse_std, r_squared):
    #Writing the output file
    os.chdir("./Results")
    f= open("output.txt", "a")
    f.write("Feature dimension: " + str(feature_dimension)+"\n")
    f.write("Average MSE: " + str(round(average_mse,4))+ "\tAverage STD: "+ str(round(mse_std,4))+ "\t R^2: "+ str(r_squared)+"\n")
    os.chdir("..")
    f.close()

if __name__ == "__main__":
    filepath = "./Data/"

    # User defined : (The student should find out an appropirate dimension. (it is integer (1-9).))
    feature_dimension = 3

    # For evaluation
    mse_list = []

    for i in range(10):
        # Generation filepath
        train_x, train_y, test_x, test_y = filename_generation(filepath, i)

        # File import
        train_x_data, train_y_data = create_data_np(train_x, train_y)
        test_x_data, test_y_data = create_data_np(test_x, test_y)

        # Feature engineering
        feature = PolynomialFeatures(feature_dimension)
        X_train = feature.transform(train_x_data)
        X_test = feature.transform(test_x_data)
        
        #For plotting
        x_test_plot = np.linspace(0, 1, 100)
        X_test_plot = feature.transform(x_test_plot)
        
        # Fit
        model = LinearRegressor()

        # Training (Learning)
        model.fit(X_train, train_y_data)

        # Predicting
        y_hat = model.predict(X_test)
        y_hat_plot = model.predict(X_test_plot)

        # Evaluating
        mse_value = mse(test_y_data, y_hat)

        # Appending
        mse_list.append(mse_value)

        # Plotting
        plot(train_x_data, train_y_data, test_x_data, test_y_data, x_test_plot, y_hat_plot, "result_{}".format(i))
        


    # average MSE 
    (average_mse, mse_std) = average_mse(mse_list)

    # Print
    print(
        '[Average MSE] \n',
        '{:7.4f} \n'.format(average_mse),
        '[MSE_std] \n',
        '{:7.4f}'.format(mse_std))
    
    y_var = np.var(test_y_data)
    r_squared = 1- average_mse/y_var
    r_squared = round(r_squared,4)

    #write_file(feature_dimension,average_mse,mse_std, r_squared)

    

    
    

        
        

    
    
    

    

    

    
    
    

    