import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import matplotlib.pyplot as plt
import numpy as np
import itertools
import sho
import qtt_utility as ut
import linear as lin
import differential as df
import Ex_sin as ss
import MPSUtility as mpsut
from scipy.sparse.linalg import LinearOperator, eigsh, eigs
import dmrg


def toUniTen (T):
    T = cytnx.from_numpy(T)
    return cytnx.UniTensor (T)

def get_ele (qtt, inds):
    res = qtt[0].get_block().numpy()[:,inds[0],:]
    N = len(qtt)
    for n in range(1,N):
        M = qtt[n].get_block().numpy()
        M = M[:,inds[n],:]
        res = np.dot(res, M)
    return res[0,0]


N = 10
rescale = 0.01

# Define the MPO
H, L, R = sho.get_H_SHO (N, rescale)
for i in range(len(H)):
    H[i] = toUniTen(H[i])
    H[i].relabels_(['l','ip','i','r'])

L = L.reshape((7,1,1))
R = R.reshape((7,1,1))
L0 = toUniTen (L)
R0 = toUniTen (R)
L0.relabels_(['mid','up','dn'])
R0.relabels_(['mid','up','dn'])


# Get initial state MPS
psi = mpsut.generate_random_MPS (N, d=2, D=2)


# Define the bond dimensions for the sweeps
maxdims = [2]*10 + [4]*10 + [8]*10 + [16]*10 + [32]*30
cutoff = 1e-12

# Run dmrg
psi = dmrg.dmrg (psi, H, L0, R0, maxdims, cutoff)


maxdims = maxdims + [64]*80
psi2 = mpsut.generate_random_MPS (N, d=2, D=2)
psi2 = dmrg.dmrg (psi2, H, L0, R0, maxdims, cutoff, ortho_mpss=[psi], weights=[20])

maxdims = maxdims + [64]*256
psi3 = mpsut.generate_random_MPS (N, d=2, D=2)
psi3 = dmrg.dmrg (psi3, H, L0, R0, maxdims, cutoff, ortho_mpss=[psi,psi2], weights=[20,20])

# Write MPS
#for i in range(len(psi)):
#    psi[i].Save('A0_'+str(i))

# Plot
xs,fs,fs2,fs3 = [],[],[],[]
ranges = [range(2) for _ in range(N)]

# Use itertools.product to generate the Cartesian product of the reversed ranges
for indices in itertools.product(*ranges):
    inds = list(reversed(indices))
    x = ut.inds_to_x (inds, rescale)
    xs.append(x)
    fs.append(get_ele (psi, inds))
    fs2.append(get_ele (psi2, inds))
    fs3.append(get_ele (psi3, inds))

np.savetxt('a.dat',(xs,fs))
np.savetxt('a2.dat',(xs,fs2))
np.savetxt('a3.dat',(xs,fs3))
plt.plot (xs, fs, marker='.', ls='None')
plt.plot (xs, fs2, marker='.', ls='None')
plt.plot (xs, fs3, marker='.', ls='None')
plt.show()

