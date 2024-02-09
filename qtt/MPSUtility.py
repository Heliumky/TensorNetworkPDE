import os, sys
sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import numpy as np

def generate_random_MPS (Nsites, d, D):
    psi = [None for i in range(Nsites)]
    for i in range(Nsites):
        if i == 0:
            A = np.random.rand(1,d,D)
        elif i == (Nsites-1):
            A = np.random.rand(D,d,1)
        else:
            A = np.random.rand(D,d,D)
        A = cytnx.from_numpy(A)
        psi[i] = cytnx.UniTensor (A, rowrank=2)
        psi[i].set_labels(['l','i','r'])
    return psi

#LR[0]:        LR[1]:            LR[2]:
#
#   -----      -----A[0]---     -----A[1]---
#   |          |     |          |     |
#  ML----     LR[0]--M-----    LR[1]--M-----      ......
#   |          |     |          |     |
#   -----      -----A*[0]--     -----A*[1]--
#
#
class LR_envir_tensors_mpo:
    def __init__ (self, N, L0, R0):
        # Network for computing right environment tensors
        self.R_env_net = cytnx.Network()
        self.R_env_net.FromString(["R: r, rup, rdn",
                                   "B: ldn, i, rdn",
                                   "M: l, r, ip, i",
                                   "B_Conj: lup, ip, rup",
                                   "TOUT: l, lup, ldn"])
        # Network for computing left environment tensors
        self.L_env_net = cytnx.Network()
        self.L_env_net.FromString(["L: lup, l, ldn",\
                                   "A: ldn, i, rdn",\
                                   "M: l, ip, r, i",\
                                   "A_Conj: lup, ip, rup",\
                                   "TOUT: r, rup, rdn"])
        self.centerL = 0
        self.centerR = N-1
        self.LR = [None for i in range(N+1)]
        self.LR[0]  = L0
        self.LR[-1] = R0

    def __getitem__(self, i):
        return self.LR[i]

    def update_LR (self, mps1, mps2, mpo, center):
        # Update the left environments
        for p in range(self.centerL, center):
            # Set the network for the left environment tensor on the current site
            self.L_env_net.PutUniTensor("L", self.LR[p], ['up','mid','dn'])
            self.L_env_net.PutUniTensor("A", mps1[p], ['l','i','r'])
            self.L_env_net.PutUniTensor("M", mpo[p], ['l','ip','r','i'])
            self.L_env_net.PutUniTensor("A_Conj", mps2[p], ['l','i','r'])
            self.LR[p+1] = self.L_env_net.Launch()
            self.LR[p+1].relabels_(['mid','up','dn'])
        # Update the right environments
        for p in range(self.centerR, center, -1):
            # Set the network for the right environment tensor on the current site
            self.R_env_net.PutUniTensor("R", self.LR[p+1], ['mid','up','dn'])
            self.R_env_net.PutUniTensor("B", mps1[p], ['l','i','r'])
            self.R_env_net.PutUniTensor("M", mpo[p], ['l','r','ip','i'])
            self.R_env_net.PutUniTensor("B_Conj", mps2[p], ['l','i','r'])
            self.LR[p] = self.R_env_net.Launch()
            self.LR[p].relabels_(['mid','up','dn'])
        self.centerL = self.centerR = center

class LR_envir_tensors_mps:
    def __init__ (self, N):
        # Network for computing right environment tensors
        self.env_net = cytnx.Network()
        self.env_net.FromString(["LR: _up, _dn",
                                 "A: _dn, i, dn",
                                 "Adag: _up, i, up",
                                 "TOUT: up, dn"])
        self.centerL = 0
        self.centerR = N-1
        self.LR = [None for i in range(N+1)]
        L0 = np.array([1]).reshape((1,1))
        R0 = np.array([1]).reshape((1,1))
        L0 = cytnx.UniTensor (cytnx.from_numpy(L0))
        R0 = cytnx.UniTensor (cytnx.from_numpy(R0))
        L0.relabels_(['up','dn'])
        R0.relabels_(['up','dn'])
        self.LR[0]  = L0
        self.LR[-1] = R0

    def __getitem__(self, i):
        return self.LR[i]

    def update_LR (self, mps1, mps2, center):
        # Update the left environments
        for p in range(self.centerL, center):
            # Set the network for the left environment tensor on the current site
            self.env_net.PutUniTensor("LR", self.LR[p], ['up','dn'])
            self.env_net.PutUniTensor("A", mps1[p], ['l','i','r'])
            self.env_net.PutUniTensor("Adag", mps2[p], ['l','i','r'])
            self.LR[p+1] = self.env_net.Launch()
        # Update the right environments
        for p in range(self.centerR, center, -1):
            # Set the network for the right environment tensor on the current site
            self.env_net.PutUniTensor("LR", self.LR[p+1], ['up','dn'])
            self.env_net.PutUniTensor("A", mps1[p], ['r','i','l'])
            self.env_net.PutUniTensor("Adag", mps2[p], ['r','i','l'])
            self.LR[p] = self.env_net.Launch()
        self.centerL = self.centerR = center

