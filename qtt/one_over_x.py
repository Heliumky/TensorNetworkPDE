import numpy as np
import matplotlib.pyplot as plt
from ncon import ncon
import qtt_utility as ut
import linear as lin
import differential as df

# MPO tensor:
#                 (ipr)                   (ipr)                        (ipr)
#                   1                       0                            1
#                   |                       |                            |
#         (k1)  0 --o-- 3 (k2)              o-- 2 (k)           (k)  0 --o
#                   |                       |                            |
#                   2                       1                            2
#                  (i)                     (i)                          (i)
#
#
#                   2                       0                            2
#                   |                       |                            |
#           T1  0 --o-- 4                   o-- 2                    0 --o
#                   |                       |                            |
#           T2  1 --o-- 5                   o-- 3                    1 --o
#                   |                       |                            |
#                   3                       1                            3
#
#
#                   1                       0                           1
#                   |                       |                           |
#               0 --o-- 2                   o-- 1                   0 --o
#


#get the dimension of each tensor legs.
#implement the multiplication of mpo.
#the properties of each mpo is (k1,ipr,i,k2) 
def prod_mpo_tensor (T1, T2):
    di = T2.shape[2]    #get the dimension of physical "i" legs.
    dipr = T1.shape[1]  #get the dimension of physical "ipr" legs.
    dk1 = T1.shape[0] * T2.shape[0] #after the multiplication of two mpos to be a new mpo and  getten the dimension of virtual "k1" legs of new mpo.
    dk2 = T1.shape[3] * T2.shape[3] #so on.

    T = ncon ([T1,T2], ((-1,-3,1,-5), (-2,1,-4,-6)))    #contract the physical bond between the "i" and "ipr"
    T = T.reshape ((dk1,dipr,di,dk2))
    return T



#implement the addition of mpos.
def sum_mpo_tensor (T1, T2):
    res = np.zeros((T1.shape[0]+T2.shape[0], T1.shape[1], T1.shape[2], T1.shape[3]+T2.shape[3])) #virtual bound T[0], T[3] adding and physical T1[1] and T1[2].
    res[:T1.shape[0],:,:,:T1.shape[3]] = T1 #be careful the syntax of [:T1.shape[0]...] it is 4 dim tensor. 
    res[T1.shape[0]:,:,:,T1.shape[3]:] = T2 #be careful the syntax of [T1.shape[0]:,...] it is 4 dim tensor. 
    return res

def get_H_inversef (N, rescale):
    xmax = rescale * (2**N-1)
    shift = -0.01
    print('xmax =',xmax)
    print('xshift =',shift)

    H = []
    for n in range(N):
        x_tensor = lin.make_x_tensor (n, rescale)
        ddx2_tensor = df.make_tensorA()
        ddx2f_tensor = prod_mpo_tensor (ddx2_tensor, x_tensor)
        inverse_f_tensor = prod_mpo_tensor (x_tensor.T, ddx2f_tensor)
        H.append(inverse_f_tensor)

    L_x, R_x = lin.make_x_LR (shift)
    L_ddx2, R_ddx2 = df.make_LR()
    L_ddx2x = ncon([L_ddx2,L_x], ((-1,),(-2,))).reshape(-1,)
    R_ddx2x = ncon([R_ddx2,R_x], ((-1,),(-2,))).reshape(-1,)
    L = ncon([L_x.T,L_ddx2x], ((-1,),(-2,))).reshape(-1,)
    R = ncon([R_x.T,R_ddx2x], ((-1,),(-2,))).reshape(-1,)
    return H, L, R