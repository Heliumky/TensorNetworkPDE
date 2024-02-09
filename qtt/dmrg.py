#import os, sys
#os.environ["OMP_NUM_THREADS"] = "4"
#sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import numpy as np
import MPSUtility as mpsut
from scipy.sparse.linalg import LinearOperator, eigsh, eigs


# An effective Hamiltonian must:
# 1. Inherit <cytnx.LinOp> class
# 2. Has a function <matvec> that implements H|psi> operation

class eff_Hamilt (cytnx.LinOp):
    def __init__ (self, L, M1, M2, R):
        # Initialization
        cytnx.LinOp.__init__(self,"mv", 0)

        # Define network for H|psi> operation
        self.anet = cytnx.Network()
        self.anet.FromString(["psi: ldn, i1, i2, rdn",
                         "L: l, ldn, lup",
                         "R: r, rdn, rup",
                         "M1: l, i1, ip1, mid",
                         "M2: mid, ip2, i2, r",
                         "TOUT: lup, ip1, ip2, rup"])
        self.anet.PutUniTensor("L", L, ["mid","dn","up"])
        self.anet.PutUniTensor("M1", M1, ["l","i","ip","r"])
        self.anet.PutUniTensor("M2", M2, ["l","ip","i","r"])
        self.anet.PutUniTensor("R", R, ["mid","dn","up"])

        self.anet2 = cytnx.Network()
        self.anet2.FromString(["A1: lup, i1, _",
                               "A2: _, i2, rup",
                               "L: ldn, lup",
                               "R: rdn, rup",
                               "TOUT: ldn, i1, i2, rdn"])
        self.ortho = []
        self.ortho_w = []

    def add_orthogonal (self, L, orthoA1, orthoA2, R, weight):
        self.anet2.PutUniTensor("L", L, ["dn","up"])
        self.anet2.PutUniTensor("R", R, ["dn","up"])
        self.anet2.PutUniTensor("A1", orthoA1, ["l","i","r"])
        self.anet2.PutUniTensor("A2", orthoA2, ["l","i","r"])
        out = self.anet2.Launch()
        out.relabels_(['l','i1','i2','r'])
        self.ortho.append(out)
        self.ortho_w.append(weight)

    # Define H|psi> operation
    def matvec (self, v):
        self.anet.PutUniTensor("psi",v,['l','i1','i2','r'])
        out = self.anet.Launch()
        out.set_labels(['l','i1','i2','r'])   # Make sure the input label match output label

        for j in range(len(self.ortho)):
            ortho = self.ortho[j]
            overlap = cytnx.Contract (ortho, v)
            out += self.ortho_w[j] * overlap.item() * ortho

        return out

def dmrg (psi, H, L0, R0, maxdims, cutoff, ortho_mpss=[], weights=[]):
    # Define the links to update for a sweep
    # First do a right-to-left and then a left-to-right sweep
    Nsites = len(psi)
    ranges = [range(Nsites-2,-1,-1), range(Nsites-1)]

    # For printing information
    verbose = ["[r->l]", "[l->r]"]


    # Get the environment tensors
    LR = mpsut.LR_envir_tensors_mpo (Nsites, L0, R0)
    LR.update_LR (psi, psi, H, Nsites-1)

    LR_ortho = []
    for omps in ortho_mpss:
        lr = mpsut.LR_envir_tensors_mps (Nsites)
        lr.update_LR (psi, omps, Nsites-1)
        LR_ortho.append (lr)
    
    ens = []
    for k in range(len(maxdims)):                                                            # For each sweep
        chi = maxdims[k]                                                                     # Read bond dimension
        print ('Sweep',k,', chi='+str(chi))
        for lr in [0,1]:
            for p in ranges[lr]:                                                             # For each bond
                M1, M2 = H[p], H[p+1]
                # Compute the current psi
                A1 = psi[p].relabels(['l','i','r'], ['l','i1','_'])
                A2 = psi[p+1].relabels(['l','i','r'],['_','i2','r'])
                phi = cytnx.Contract (A1, A2)

                # Define the effective Hamiltonian
                effH = eff_Hamilt (LR[p], M1, M2, LR[p+2])

                # orthogonal MPS
                for j in range(len(ortho_mpss)):
                    omps = ortho_mpss[j]
                    weight = weights[j]
                    oLR = LR_ortho[j]
                    effH.add_orthogonal (oLR[p], omps[p], omps[p+1], oLR[p+2], weight)

                # Find the ground state for the current bond
                enT, phi = cytnx.linalg.Lanczos (effH, phi, method = "Gnd", CvgCrit=999, Maxiter=2)
                en = enT.item()                                                                 # Tensor to number

                # Store the energy
                ens.append(en);

                # SVD and truncate the wavefunction psi
                phi.set_rowrank_(2)
                s, u, vt = cytnx.linalg.Svd_truncate(phi, keepdim=chi, err=cutoff)      

                # Setting psi[p] = u, psi[p+1] = vt
                psi[p] = u
                psi[p+1] = vt
                
                # Normalize the singular values
                s = s/s.get_block_().Norm().item()

                if lr == 0:
                    # Absorb s into next neighbor
                    psi[p] = cytnx.Contract(psi[p],s)

                    psi[p].relabels_(['l', 'i1', '_aux_R'],['l','i','r'])
                    psi[p+1].relabels_(['_aux_R', 'i2', 'r'],['l','i','r'])

                    LR.update_LR (psi, psi, H, p)
                    for j in range(len(ortho_mpss)):
                        LR_ortho[j].update_LR (psi, ortho_mpss[j], p)
                if lr == 1:
                    # Absorb s into next neighbor
                    psi[p+1] = cytnx.Contract(s,psi[p+1])

                    psi[p].relabels_(['l', 'i1', '_aux_L'],['l','i','r'])
                    psi[p+1].relabels_(['_aux_L', 'i2', 'r'],['l','i','r'])

                    LR.update_LR (psi, psi, H, p+1)
                    for j in range(len(ortho_mpss)):
                        LR_ortho[j].update_LR (psi, ortho_mpss[j], p+1)
                
            print ('\t',verbose[lr],'energy =',en)
    return psi

