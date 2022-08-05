import numpy as np
import itertools as it

def uni2x2(angles,i,j,d):
    """
    unitary of shape exp(i gamma sigma_x) exp(i delta sigma_z) on 2x2
    subspace of a dxd matrix
    """
    U = np.zeros([d,d],dtype=np.complex);
    U[i][i] = np.cos(angles[0])#*np.exp(1j*angles[1]);
    U[j][j] = np.conjugate(U[i][i]);
    U[i][j] = np.sin(angles[0])*1j*np.exp(-1j*angles[1]);
    U[j][i] = np.sin(angles[0])*1j*np.exp(1j*angles[1]);
    for x in [y for y in range(d) if y!=i and y!=j]:
        U[x][x] = 1
    return U


def mdot(systems):
    """
    args:       systems : array_like, ndarray or list
    returns:    ndarray
    """
    
    k = 0
    params = []
    for s in systems:
        params.append(s)
        params.append([k, k+1])
        k+=1
        
    return np.einsum(*params)


def dagger(M):
    return M.conj().T


def unitarynxn(n,pars):
    res = False
    if len(pars)==n**2:
        phases = pars[0:n]
        phmat = np.diag([np.exp(1j*phi) for phi in phases])
        i = n
        Ul = []
        for sub in it.combinations(range(n),2):
            u = uni2x2(pars[i:i+2],sub[0],sub[1],n)
            Ul.append(u)
            i = i+2
        U = mdot(Ul)
        res = np.dot(U,phmat)
    return res
    

def test():
    A = np.random.random([2,2])
    B = np.random.random([2,2])
    C = np.random.random([2,2])
   #print(np.dot(B,np.dot(A,np.dot(B,C))))
   #print(mdot([B,A,B,C]))

          
    U = unitarynxn(3,np.random.random(9))
    print('U--------------------')
    print(U)
    print('id?--------------------')
    print(np.dot(dagger(U),U))
    return 0

#test()
            
  
