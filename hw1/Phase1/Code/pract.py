import numpy as np


a = np.arange(8).reshape((2,2,2))
b = np.array([[0,0],[1,1]])

print(a[0,:,:])
print(b)
print(a[b[:,0],b[:,1],:])

