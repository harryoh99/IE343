import numpy as np

import matplotlib.pyplot as plt


def create_data_np(filepath1, filepath2):
    
    x = np.genfromtxt(filepath1, dtype=None, encoding='UTF-8')
    t = np.genfromtxt(filepath2, dtype=None, encoding='UTF-8')

    return x, np.int_(t)

def filename_generation(path, i):
    train_x_filepath = path + "x_{}.csv".format(i) 
    train_y_filepath = path + "y_{}.csv".format(i)
    test_x_filepath = path + "test_x_{}.csv".format(i)
    test_y_filepath = path + "test_y_{}.csv".format(i)

    return train_x_filepath, train_y_filepath, test_x_filepath, test_y_filepath



