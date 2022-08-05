""" Improved version of violationseesaw, handle arbitrary number of parties and
measurement settings and system dimensions. Only dichotomic measurements
"""
import numpy as np
import picos as pic
import cvxopt as cvx
import bellnp as bn
import itertools as it


def randomunitary(n):
    M = (np.random.randn(n,n) + 1j*np.random.randn(n,n))/np.sqrt(2);
    q,r = np.linalg.qr(M);
    d = np.diagonal(r);
    d = d/np.absolute(d);
    q = np.multiply(q,d,q);
    return q

def dagger(v):
    return v.conjugate().transpose();

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

def opt_meas(sys, BI, rho, otherpovms): 
    # arbitrary number of parties with arbitrary dim, 2 outcomes per measurement
    # pass all povms, also the ones are going to be updated in this run
    dims = [ np.shape(otherpovms[i][0][0])[0] for i in range(len(otherpovms)) ]
    d = dims[sys]
    nmeas = np.array(np.shape(BI)) - 1
    P = [[pic.HermitianVariable('P{}{}'.format(x,a), [d, d]) for a in
range(2)] for x in range(nmeas[sys])]    
    povms = otherpovms[:]
    povms[sys] = P
    obs = [[ sett[0] - sett[1] for sett in party] for party in povms]
    obs = [ [np.eye(dims[i])] + obs[i]  for i in range(len(obs)) ]
    corrs = [ pickron(x) for x in it.product(*obs) ]
    B = BI.flatten()
    BO = sum([B[i]*corrs[i] for i in range(len(B))])
    Pr = pic.Problem()
    Pr.set_objective('min', ((rho|BO) + (BO|rho))/2)

    Pr.add_list_of_constraints([pic.sum([P[x][a] for a in range(2)]) ==np.eye(d) \
 for x in range(nmeas[sys])],key="probabilities sum up to 1")
    Pr.add_list_of_constraints([P[x][a] >> 0 for a in range(2) \
 for x in range(nmeas[sys])],key="effects are PSD")

    S = Pr.solve(solver='mosek',verbose=0,tol=1e-7)
   #print(S)
    
    return [[np.array(cvx.matrix(x.value)) for x in sett] for sett in P], S.value

povms2obs = lambda L: [[x[0] - x[1] for x in P] for P in L]

def obs2povm(obs):
    lamda, V = np.linalg.eig(obs)
    order = np.argsort(lamda)[::-1]
    povm = [np.outer(V[:, o], V[:, o].conj()) for o in order]
    return povm

obs2povms = lambda L: [[obs2povm(O) for O in party] for party in L]

def opt_state(povms,B):
    dims = [ np.shape(povms[i][0][0])[0] for i in range(len(povms)) ]
    meas = [[x[0] - x[1] for x in P] for P in povms]
    Bo = bn.belloperator(B,meas)


    lamda, V = np.linalg.eig(Bo)
    ind = np.argmin(lamda)
    v = V[:, ind]
    rho = np.outer(v, v.conj())
    viol = lamda[ind]
    
    return rho, viol




##  def opt_state_sdp(povms,B):
##      dims = [ np.shape(povms[i][0][0])[0] for i in range(len(povms)) ]
##      meas = [[x[0] - x[1] for x in P] for P in povms]
##      Bo = bn.belloperator(B,meas)

##      Pr = pic.Problem()
##      rho = pic.HermitianVariable('rho', (np.product(dims),np.product(dims)))
##      Pr.add_constraint(rho >> 0)
##      Pr.add_constraint(pic.trace(rho) == 1)
##      Pr.set_objective('min',((Bo|rho) + (rho|Bo))/2)
##      S = Pr.solve(solver='mosek',tol=1e-6,verbose=0)
##     #print('value', S.value)
##      ret = cvx.matrix(rho.value)
##      return ret, S.value

def randompovm(d=2):
    # 2 outcomes
    U = randomunitary(d) 
    norm = lambda x: x/np.sqrt(np.dot(x,x))
    diag = np.diag(norm(np.absolute(np.random.randn(d))))
    uni = lambda u,x: np.einsum('ij,jk,kl->il',u,x,dagger(u))
    P = [uni(U,diag), np.eye(d) - uni(U,diag)] 
    return P

def randpovmsforscenario(nmeas, dims):
    return [[randompovm(dims[i]) for j in range(nmeas[i])] for i in
range(len(nmeas))]


def simple_violation(B, ipovms, repetitions=10,verbose=False):
   #print('sv started')
    povms = list(ipovms[:])
    nparties = len(np.shape(B))
    for n in range(repetitions):
        rho, viol = opt_state(povms, B)
        if verbose:
            print('simple viol', viol)
        for i in range(nparties):
            P, viol = opt_meas(i, B, rho, povms)
            povms[i] = P
    return viol, rho, povms

def violation(B, dims, repetitions=17, initvals=15, initrep=7, numtol=1e-6,verbose=False):
    nmeas = list(np.array(np.shape(B)) - 1)
    startpovms = [randpovmsforscenario(nmeas, dims) for i in range(initvals)]
    if verbose:
        print('checking out init vals')
#   pool = mp.Pool(mp.cpu_count())
#   results = [pool.apply(simple_violation, args = (B, x), kwds =
#{"repetitions":initrep, "verbose":verbose}) for x in startpovms]
#   pool.close()
    results = [simple_violation(B, x, repetitions=initrep, verbose=verbose) for x in startpovms]
    beststart = np.argmin([x[0] for x in results])
    povms = results[beststart][2]
    rep = 0
    oldviol = 100
    improv = 100
    if verbose:
        print('final tweaking')
    while improv>numtol or rep<repetitions-initrep:
        rep = rep+1
        rho, viol = opt_state(povms, B)
        if verbose:
            print('viol', viol)
        for i in range(len(nmeas)):
            P, viol = opt_meas(i, B, rho, povms)
            povms[i] = P
        improv = oldviol - viol
        oldviol = viol
    return viol, rho, povms


#   def test():
#       bellfile = "all_i3322_gen_wo_dup_sorted.list"
#       B = b3.tab_from_human_sym_file(2,bellfile)
#       print('B',B)
#       print(violation(B,[2,2,2],verbose=True)) 
#       return 0


#test()

def simple_violation_fix_state(B, ipovms, psi, repetitions=10,verbose=False):
   #print('sv started')
    rho = np.outer(psi.conj(), psi)
    povms = list(ipovms[:])
    nparties = len(np.shape(B))
    for n in range(repetitions):
        for i in range(nparties):
            P, viol = opt_meas(i, B, rho, povms)
            povms[i] = P
        if verbose:
            print('simple viol', viol)
    return viol, povms

def violation_fix_state(B, dims, psi, ipovms = None, repetitions=17, initvals=15, initrep=7, numtol=1e-6,verbose=False):
    rho = np.outer(psi.conj(), psi)
    nmeas = list(np.array(np.shape(B)) - 1)
    if ipovms is None:
        startpovms = [randpovmsforscenario(nmeas, dims) for i in range(initvals)]
    else:
        startpovms = ipovms
    if verbose:
        print('checking out init vals')
    results = [simple_violation_fix_state(B, x, psi, repetitions=initrep, verbose=verbose) for x in startpovms]
    beststart = np.argmin([x[0] for x in results])
    povms = results[beststart][1]
    rep = 0
    oldviol = 100
    improv = 100
    if verbose:
        print('final tweaking')
    while improv>numtol or rep<repetitions-initrep:
        rep = rep+1
        if verbose:
            print('viol', viol)
        for i in range(len(nmeas)):
            P, viol = opt_meas(i, B, rho, povms)
            povms[i] = P
        improv = oldviol - viol
        oldviol = viol
    return viol, povms


