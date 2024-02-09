import numpy as np
from ncon import ncon

I = np.array([[1.,0.],[0.,1.]])
sp = np.array([[0,1],[0,0]])
sm = np.array([[0,0],[1,0]])


    # given the binary array(inds) to transfer to the decimal value (x)
def inds_to_x (inds, rescale=1., shift=0.):
    res = inds[0]
    for i in range(1,len(inds)):
        res += inds[i] * 2**i
    return rescale * res + shift

def mpo_to_mps (qtt):
    p = np.zeros((2,2,2))
    p[0,0,0] = 1.
    p[1,1,1] = 1.
    # Reduce to matrix multiplication
    Ms = []                 # Store the matrices after index collapsing
    N = len(qtt)            # The number of tensors in QTT
    for n in range(N):      # For each tensor
        M = qtt[n]
        M = ncon ([M,p], ((-1,1,2,-3), (1,2,-2)))
        Ms.append(M)
    return Ms


# the function value was getten from the function mps(qqt of function)
def get_ele_func (qtt, L, R, inds):
    # Reduce to matrix multiplication
    N = len(qtt)            # The number of tensors in QTT
    res = L
    for n in range(N):      # For each tensor
        ind = inds[n]       # The index number we want to collapse
        M = qtt[n][:,ind,:]
        res = res @ M
    res = res @ R
    return res    

def get_ele_op (optt, inds):
    N = len(optt)            # The number of tensors in QTT
    res = 1.
    for n in range(len(optt)):
        M = optt[n]
        if n == 0:
            M = M[inds[n], inds[n], :]
        elif n == N - 1:
            M = M[:, inds[n], inds[n]]
        else:
            M = M[:, inds[n], inds[n], :]

        res = np.dot(res, M)
    return res

