import numpy as np
import matplotlib.pyplot as plt

def plot(x_train, y_train, x_test, y_test, x1_test, x2_test, model, filename):
    # Training Data
    plt.subplot(1,2,1)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=50, alpha=0.5, label="training data")
    
    # Plot support vectors
    plt.scatter(model.X[:, 0], model.X[:, 1], s=100, facecolor="none", edgecolor="g")
    plt.xlim(-5, 15)
    plt.ylim(-5, 15)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()

    # Test Data
    plt.subplot(1,2,2)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=50, alpha=0.5, label="test data")
    x = np.array([x1_test, x2_test]).reshape(2, -1).T
    plt.contourf(x1_test, x2_test, model.distance(x).reshape(100, 100), alpha=0.1, levels=np.linspace(-1, 1, 3))
    
    # plot property
    plt.xlim(-5, 15)
    plt.ylim(-5, 15)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    #plt.show()
    plt.savefig(filename + ".png")
    plt.clf()

def plot_2(x_train, y_train, x_test, y_test, x1_test, x2_test, model, filename):
    # Training Data
    plt.subplot(1,2,1)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=50, alpha=0.5, label="training data")
    
    # Plot support vectors
    plt.scatter(model.X[:, 0], model.X[:, 1], s=100, facecolor="none", edgecolor="g")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()

    # Test Data
    plt.subplot(1,2,2)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=50, alpha=0.5, label="test data")
    x = np.array([x1_test, x2_test]).reshape(2, -1).T
    plt.contourf(x1_test, x2_test, model.distance(x).reshape(100, 100), alpha=0.1, levels=np.linspace(-1, 1, 3))
    
    # plot property
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
   # plt.show()
    plt.savefig(filename + ".png")
    plt.clf()
