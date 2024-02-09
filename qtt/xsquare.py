#!/usr/bin/env python
# coding: utf-8

#sin(t_{1} + t_{2} + ... + t_{N}) = [sin(t_0), cos(t_0)] * [[cos(t_{i}),-sin(t_i)],[sin(t_i),cos(t_i)]] * [cos(t_N), sin(t_N)]

import numpy as np
import matplotlib.pyplot as plt


# factor is the difference between x_i to x_{i+1},
# the inds[i] is the binary_arr[i], e.g inds = [1,0,1]. x= 1*2**0 + 0*2**1 + 1*2**2
def inds_to_x (inds, factor=1.):
    res = inds[0]
    for i in range(1,len(inds)):
        res += inds[i] * factor * 2**i
    return res

# create the most left part of mps. 
def make_tensorL (n, factor=1.):
    AL = np.zeros ((2,3))  # (i, k) #Lmps(left mps) [t_0**2, 2*t_0, 1], "i" is physical bond, 
    x = factor * 2**n      # k is virtual(auxiliary) bond, here t_k reprenst x. t_m = a/d + 2**m*binary_arr[m]*h,
    # k == 0               # here, setting a = 0. In physical bond, the [[0],[1]] is x=0 and [[1],[0]] is the x=1.
    AL[0,0] = 0            # index k conctrol sin or cos, k=0 correspond to sin, k=1 to sin. i=0 is x = 2**0*(binary_arr[0]=0)*(factor=1),
    AL[1,0] = x**2    # i = 1 is x = 2**0*(binary_arr[0]=1)*(factor=1),
    # k == 1
    AL[0,1] = 0
    AL[1,1] = 2*x
    # k == 2
    AL[0,2] = 1
    AL[1,2] = 1
    return AL

def make_tensorR (n, factor=1.):
    AR = np.zeros ((3,2))  # (k, i)  #Rmps(right mps) [1, t_0, t_0**2], "i" is physical bond, 
    x = factor * 2**n      # k is virtual(auxiliary) bond, here t_k reprenst x. t_m = a/d + 2**m*binary_arr[m]*h,
    # k == 0               # here, setting a = 0. In physical bond, the [[0],[1]] is x=0 and [[1],[0]] is the x=1.
    AR[0,0] = 1            # index k conctrol sin or cos, k=0 correspond to cos, k=1 to cos. i=0 is x = 2**0*(binary_arr[0]=0)*(factor=1),
    AR[0,1] = 1    # i = 1 is x = 2**0*(binary_arr[0]=1)*(factor=1),
    # k == 1
    AR[1,0] = 0
    AR[1,1] = x
    # k == 2
    AR[2,0] = 0
    AR[2,1] = x**2
    return AR

def make_tensorA (n, factor=1.):
    A = np.zeros ((3,2,3)) # (k1,i,k2) mps(bulk mps), "i" is physical bond, k1/k2 is left/right virtual bond.
    x = factor * 2**n      # the bulk mps is [[cos(t_{i}),-sin(t_i)],[sin(t_i),cos(t_i)]]
    # k1 == 0, k2 == 0
    A[0,0,0] = 1           # cos (t_i = 0) = 1
    A[0,1,0] = 1   # cos (t_i = x)
    # k1 == 1, k2 == 0
    A[1,0,0] = 0           # sin (t_i = 0) = 0
    A[1,1,0] = x   # sin (t_i = x) = x
    # k1 == 2, k2 == 0
    A[2,0,0] = 0           # sin (t_i = 0) = 0
    A[2,1,0] = x**2   # sin (t_i = x) = x

    # k1 == 0, k2 == 1
    A[0,0,1] = 0           # cos (t_i = 0) = 1
    A[0,1,1] = 0   # cos (t_i = x)
    # k1 == 1, k2 == 1
    A[1,0,1] = 1           # sin (t_i = 0) = 0
    A[1,1,1] = 1   # sin (t_i = x) = x
    # k1 == 2, k2 == 1
    A[2,0,1] = 0           # sin (t_i = 0) = 0
    A[2,1,1] = 2*x   # sin (t_i = x) = x

    # k1 == 0, k2 == 2
    A[0,0,2] = 0           # cos (t_i = 0) = 1
    A[0,1,2] = 0   # cos (t_i = x)
    # k1 == 1, k2 == 2
    A[1,0,2] = 0           # sin (t_i = 0) = 0
    A[1,1,2] = 0   # sin (t_i = x) = x
    # k1 == 2, k2 == 2
    A[2,0,2] = 1           # sin (t_i = 0) = 0
    A[2,1,2] = 1   # sin (t_i = x) = x
    return A


def make_xsq_qtt (N, factor):
    qtt = []        # Quantic tensor train, bult up array to store the mps of sine function.
    for n in range(N):
        if n == 0:
            A = make_tensorL(n, factor)
        elif n == N-1:
            A = make_tensorR(n, factor)
        else:
            A = make_tensorA(n, factor)
        qtt.append(A)
    return qtt

def make_sin_cos_op (N, rescale):
    qtt = []        # Quantic tensor train
    for n in range(N):
        A = make_tensorA(n, rescale)    # the rescale means that change the difference of x_i between the x_{i+1}.
        qtt.append(A)
    return qtt


""" def get_ele (qtt, inds):
    # Reduce to matrix multiplication
    Ms = []                 # Store the matrices after index collapsing
    N = len(qtt)            # The number of tensors in QTT(contain the left/right/bulk tensors.)
    for n in range(N):      # For each tensor
        ind = inds[n]       # The index number we want to collapse, the 

        M = qtt[n]
        if n == 0:          # The most left tensor
            M = M[ind,:]
        elif n == N-1:      # The most right tensor
            M = M[:,ind]
        else:               # The bulk tensors
            M = M[:,ind,:]
        Ms.append(M)
        #print(n,'M =\n',M) """

def get_ele(qtt, inds):
    res = np.eye(3)

    N = len(qtt)

    for n in range(N):
        M = qtt[n]

        if n == 0:
            M = M[inds[n], :]   # (i, k), the i(physical bond was decide by inds[i])
        elif n == N - 1:
            M = M[:, inds[n]]   # (k, i)
        else:
            M = M[:, inds[n], :]    # (k1, i, k2)
        res = np.dot(res, M)
    return res

if __name__ == '__main__':
    N = 10
    factor = 1
    qtt = make_sin_qtt (N, factor)

    inds = []
    xs,fs = [],[]
    for i10 in [0,1]:
     for i9 in [0,1]:
      for i8 in [0,1]:
       for i7 in [0,1]:
        for i6 in [0,1]:
         for i5 in [0,1]:
          for i4 in [0,1]:
           for i3 in [0,1]:
            for i2 in [0,1]:
                for i1 in [0,1]:
                    inds = [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10]
                    x = inds_to_x (inds)
                    a = get_ele (qtt, inds)
                    print(inds, x, a)
                    #print('-----------------------------------')
                    xs.append(x)
                    fs.append(a)
    plt.plot (xs, fs, 'x')
    plt.plot (xs, np.array(xs)**2, 'o')
    plt.show()





