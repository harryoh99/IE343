import numpy as np
import matplotlib.pyplot as plt

from App.data_import import *

from App.evaluation import *
from App.plot import *

# Kernel
from App.polynomial import *
from App.rbf import *
from App.euclidean import *

from App.support_vector_classifier import *


def training(kernel, is_previous_dataset=True):
    '''
        kernel : Kernel class
        filename_generation : func
    '''

    # For evaluation
    metric_list = []

    if is_previous_dataset:
        filepath = "./Data/"
    else:
        filepath = "./Data2/"
    
    # K-fold training / validation
    for i in range(0,10):
        # Generation filepath
        train_x, train_y, test_x, test_y = filename_generation(filepath, i)

        # File import
        train_x_data, train_y_data = create_data_np(train_x, train_y)
        test_x_data, test_y_data = create_data_np(test_x, test_y)

        if is_previous_dataset:
            x1_test, x2_test = np.meshgrid(np.linspace(-5, 15, 100), np.linspace(-5, 15, 100))
        else:
             x1_test, x2_test = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))

        # Train SVM
        model = SupportVectorClassifier(kernel, C=1.5)
        print("="*25, "Starting learning parameters of SVM for dataset {}.".format(i), "="*25)
        model.fit(train_x_data, train_y_data)

        # Predicting
        y_hat = model.predict(test_x_data)
        #print(test_y_data,y_hat)
        # BCE
        accruacy_value = accruacy(test_y_data, y_hat)
        metric_list.append(accruacy_value)

        # Plotting
        
        if is_previous_dataset:
            if i >=5:
                plot(train_x_data, train_y_data, test_x_data, test_y_data, x1_test, x2_test, model, "./Results/svm_result_dataset_1_{}".format(i))
            else:
                plot_2(train_x_data, train_y_data, test_x_data, test_y_data, x1_test, x2_test, model, "./Results/svm_result_dataset_1_{}".format(i))
        else:
            plot_2(train_x_data, train_y_data, test_x_data, test_y_data, x1_test, x2_test, model, "./Results/svm_result_dataset_2_{}".format(i)) 

    # average MSE 
    (average_accruacy, accuracy_std) = average_metric(metric_list)

    return average_accruacy, accuracy_std


if __name__ == '__main__':
    
    # Define kernel
    kernel1 = RBF(np.array([1,0.5,1.2]))
    kernel2 = PolynomialKernel(degree=1)
    kernel3 = Euclidean(8)

    # Select dataset
    is_previous_dataset =False

    # Training
    accraucy_average, accruacy_std = training(kernel1, is_previous_dataset=is_previous_dataset)

    # Printing
    print(
        '[Average accruacy] \n',
        '{:7.4f} \n'.format(accraucy_average),
        '[Accuracy_std] \n',
        '{:7.4f}'.format(accruacy_std))