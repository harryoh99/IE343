import numpy as np

from App.data_import import filename_generation, create_data_np

from App.evaluation import average_BCE, binary_cross_entropy
from App.plot import plot

from App.Pre_processing.polynomial import PolynomialFeatures
from App.fishers_linear_discriminant import FishersLinearDiscriminant


if __name__ == "__main__":
    filepath = "./Data/"

    # For evaluation
    BCE_list = []

    for i in range(5,10):
        # Generatino filepath
        train_x, train_y, test_x, test_y = filename_generation(filepath, i)

        # File import
        train_x_data, train_y_data = create_data_np(train_x, train_y)
        test_x_data, test_y_data = create_data_np(test_x, test_y)

        # No Feature engineering 
        X_train = train_x_data
        X_test = test_x_data
        
        #For plotting
        x1_test, x2_test = np.meshgrid(np.linspace(-5, 15, 100), np.linspace(-5, 15, 100))
        x_test_plot = np.array([x1_test, x2_test]).reshape(2, -1).T
        X_test_plot = x_test_plot
        
        # Fit
        model = FishersLinearDiscriminant()

        # Training (Learning)
        model.fit(X_train, train_y_data)

        # Predicting
        y_hat = model.classify(X_test)
        y_hat_plot = model.classify(X_test_plot)

        # Evaluating
        BCE_value = binary_cross_entropy(test_y_data, y_hat)

        # Appending
        BCE_list.append(BCE_value)

        # Plotting
        plot(train_x_data, train_y_data, test_x_data, test_y_data, x1_test, x2_test, y_hat_plot, "./Results/FLDA_result_{}".format(i))

    # average MSE 
    (average_BCE, BCE_std) = average_BCE(BCE_list)

    # Print
    print(
        '[Average BCE] \n',
        '{:7.4f} \n'.format(average_BCE),
        '[BCE_std] \n',
        '{:7.4f}'.format(BCE_std))