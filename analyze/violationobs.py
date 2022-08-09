import numpy as np
import picos as pic
import cvxopt as cvx
import itertools as it
import bellpy.analyze.bellnp as bn

def pickron(meas):
    M = list(meas[:])
    def mult(M):
        M[0] = pic.kron(M.pop(0), M[0])
        l = len(M)
        return l>1
    k = True
    while k:
        k = mult(M)
    return M[0]

def opt_meas(sys, BI, rho, otherobs, projective=False): 
    # arbitrary number of parties with arbitrary dim, 2 outcomes per measurement
    # pass all povms, also the ones are going to be updated in this run
    dims = [ np.shape(otherobs[i][0])[0] for i in range(len(otherobs)) ]
    d = dims[sys]
    nmeas = np.array(np.shape(BI)) - 1
    O = [pic.HermitianVariable('O{}'.format(x), [d, d]) for x in range(nmeas[sys])]    
    obs = otherobs[:]
    obs[sys] = O
    obs = [ [np.eye(dims[i])] + obs[i]  for i in range(len(obs)) ]
    corrs = [ pickron(x) for x in it.product(*obs) ]
    B = BI.flatten()
    BO = sum([B[i]*corrs[i] for i in range(len(B))])
    Pr = pic.Problem()
    Pr.set_objective('min', ((rho|BO) + (BO|rho))/2)

    Pr.add_list_of_constraints([o + np.eye(d) >> 0 for o in O])
    Pr.add_list_of_constraints([np.eye(d) - o >> 0 for o in O])
   #if d==2 and projective:
   #    Pr.add_list_of_constraints([pic.trace(o) == 0 for o in O])

    S = Pr.solve(solver='mosek',verbose=0,tol=1e-7)
   #print(S)
    
    return [np.array(cvx.matrix(o.value)) for o in O], S.value

def opt_state(obs,B):
    dims = [ np.shape(obs[i][0])[0] for i in range(len(obs)) ]
    Bo = bn.belloperator(B,obs)


    lamda, V = np.linalg.eig(Bo)
    ind = np.argmin(lamda)
    v = V[:, ind]
    rho = np.outer(v, v.conj())
    viol = lamda[ind]
    
    return rho, viol

#Z = np.array([[1, 0],[0, -1]], dtype=np.complex)
def randomobs(out=2):
    U = bn.randomunitary(out)
   #if out==2:
   #    Z = np.array([[1, 0],[0, -1]],dtype=np.complex)
   #else:
    Z = np.diag(np.random.randint(0,2,out)*2 - 1).astype(np.complex)
    return np.einsum('ij,jk,kl->il',U,Z,bn.dagger(U))

def randobsforscenario(nmeas, dims):
   #print('nmeas', nmeas)
    return [[randomobs(dims[i]) for j in range(nmeas[i])] for i in range(len(nmeas))]

def simple_violation(B, iobs, repetitions=10,verbose=False,
projective=False):
   #print('sv started')
    obs = list(iobs[:])
    nparties = len(np.shape(B))
    for n in range(repetitions):
        rho, viol = opt_state(obs, B)
        if verbose:
            print('simple viol', viol)
        for i in range(nparties):
            O, viol = opt_meas(i, B, rho, obs, projective=projective)
            obs[i] = O
    return viol, rho, obs

def violation(B, dims, repetitions=17, initvals=15, initrep=7,
numtol=1e-6,verbose=False, iobs=None, projective=False):
    nmeas = list(np.array(np.shape(B)) - 1)
    if iobs is None:
        startobs = [randobsforscenario(nmeas, dims) for i in range(initvals)]
    else:
        startobs = [iobs]
    if verbose:
        print('checking out init vals')
    results = [simple_violation(B, x, repetitions=initrep, verbose=verbose,
projective=projective) for\
x in startobs]
    beststart = np.argmin([x[0] for x in results])
    obs = results[beststart][2]
    rep = 0
    oldviol = 100
    improv = 100
    if verbose:
        print('final tweaking')
    while improv>numtol or rep<repetitions-initrep:
        rep = rep+1
        rho, viol = opt_state(obs, B)
        if verbose:
            print('viol', viol)
        for i in range(len(nmeas)):
            O, viol = opt_meas(i, B, rho, obs, projective=projective)
            obs[i] = O
        improv = oldviol - viol
        oldviol = viol
    return viol, rho, obs



def simple_violation_fix_state(B, iobs, rho, repetitions=10,verbose=False):
   #print('sv started')
   #rho = np.outer(psi.conj(), psi)
    obs = list(iobs[:])
    nparties = len(np.shape(B))
    for n in range(repetitions):
        for i in range(nparties):
            O, viol = opt_meas(i, B, rho, obs)
            obs[i] = O
        if verbose:
            print('simple viol', viol)
    return viol,obs

def violation_fix_state(B, dims, rho, iobs = None, repetitions=15, initvals=15, initrep=7, numtol=1e-7,verbose=False):
   #rho = np.outer(psi.conj(), psi)
    nmeas = list(np.array(np.shape(B)) - 1)
    if iobs is None:
        startobs = [randobsforscenario(nmeas, dims) for i in range(initvals)]
    else:
        startobs = [iobs]
    if verbose:
        print('checking out init vals')
    results = [simple_violation_fix_state(B, x, rho, repetitions=initrep,
verbose=verbose) for x in startobs]
    beststart = np.argmin([x[0] for x in results])
    obs = results[beststart][1]
    rep = 0
    oldviol = 100
    improv = 100
    if verbose:
        print('final tweaking')
    while improv>numtol or rep<repetitions-initrep:
        rep = rep+1
        for i in range(len(nmeas)):
            O, viol = opt_meas(i, B, rho, obs)
            obs[i] = O
      # rho, viol = opt_state(obs, B)
      # rho = 0.5*(rho + np.outer(psi.conj(), psi))
        improv = oldviol - viol
        oldviol = viol
        if verbose:
            print('viol', viol)
    return viol, obs

def test():
    B = 3*np.random.random([4,4,4])
    ghz = (bn.ket([0,0,0]) + bn.ket([1,1,1]))/np.sqrt(2)
    rhoghz = np.outer(ghz, ghz)
    v, o = violation_fix_state(B, [2,2,2], rhoghz, verbose=True)
    return 0
#test()
