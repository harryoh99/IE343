import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("./Results")
res = []
with open("output.txt","r") as f:
    line = None
    i=1
    while(1):
        line = f.readline()
        if(line == ''):
            break
        if(line[0] =="F"): 
            continue
        start = line.find("R^2")
        res.append([i,float(line[start+5:])])
        i+=1

res = np.array(res)
x = res[:,0]
y = res[:,1]

plt.plot(x,y)
plt.xlabel("Feature_dimension")
plt.ylabel("R^2")
plt.savefig("Data Compare.png")
os.chdir("..")