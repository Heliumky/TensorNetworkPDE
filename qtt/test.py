import numpy as np
import matplotlib.pyplot as plt
from ncon import ncon


I = np.array([[1.,0.],[0.,1.]])
sp = np.array([[0,1],[0,0]])
sm = np.array([[0,0],[1,0]])
L = np.array([-2,1,1])
A = np.zeros ((3,2,2,3)) # (k1,ipr,i,k2)
A[0,:,:,0] = I
A[1,:,:,0] = sp
A[2,:,:,0] = sm
A[1,:,:,1] = sm
A[2,:,:,2] = sp
B = ncon([L,A], ((1,), (1,-1,-2,-3)))
print(B,len(B),np.size(B))