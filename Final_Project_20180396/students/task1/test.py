import numpy as np

a = np.ones ((11,4))
b = np.ones ((11,1))
#print(b)
x = np.concatenate((b,a),axis=1)
print(x)

newtrain_X=np.hstack([np.ones([a.shape[0],1]), a])
print(newtrain_X==x)
