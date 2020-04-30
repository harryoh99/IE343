import numpy as np
import matplotlib.pyplot as plt

def func(x,i):
    temp = np.exp(- x**i)
    return 1/(1+temp)





Xtest = np.linspace(-5,5,100).reshape(-1,1)
for i in range(1,9):
    plt.figure()
    Ytest = func(Xtest,i)
    plt.plot(Xtest,Ytest)
    plt.ylim(0,1)
    plt.savefig("./Results/sigmoid/sig{}".format(i))


