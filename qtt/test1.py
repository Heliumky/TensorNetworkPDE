import numpy as np

A = np.array([i for i in range(18)]).reshape((3,3,2))
#eigs, vecs  = np.linalg.eigh(A)
#B = A - eigs[0]*np.dot(vecs[:,0],vecs[0])
#eigs, vecs  = np.linalg.eigh(B)
B = A[:2,:,:]
#B = A[:,0]
print(B)