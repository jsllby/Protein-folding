import numpy as np

phi = lambda x: x.astype(np.float32, copy=False)
a = np.zeros((2,2))
b = a.reshape((1,-1))

print(a.reshape((1,-1)).squeeze())
