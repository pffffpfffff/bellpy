from scipy.stats import unitary_group
import numpy
from functools import reduce
from scipy.linalg import eigh


def randomunitary(n):
    return unitary_group.rvs(n)

def dagger(v):
    return v.conjugate().transpose()

def pickron(meas):
    Ms = list(meas[:])
    return reduce(numpy.kron,Ms,numpy.array([[1]]))

def opt_meas(sys, BI, rho, otherpovms): 
    # arbitrary number of parties with arbitrary dim, 2 outcomes per measurement
    # pass all povms, also the ones are going to be updated in this run
    dims = [ np.shape(otherpovms[i][0][0])[0] for i in range(len(otherpovms)) ]
    d = dims[sys] 
    outcs = [ len(otherpovms[i][0]) for i in range(len(otherpovms)) ] # all the measurements for the same party should have the same number of outcomes
    k = outcs[sys] # k is the number of outcomes for the party sys
    nmeas = np.array(np.shape(BI)) - 1
    P = [[pic.HermitianVariable('P{}{}'.format(x,a), [d, d]) for a in
range(k)] for x in range(nmeas[sys])]    
    povms = otherpovms[:]
    povms[sys] = P
#########################################################################    
    obs = [[ sett[0] - sett[1] for sett in party] for party in povms]
    obs = [ [np.eye(dims[i])] + obs[i]  for i in range(len(obs)) ]
    corrs = [ pickron(x) for x in it.product(*obs) ] 
    B = BI.flatten()
    BO = sum([B[i]*corrs[i] for i in range(len(B))])    
    # exact information of BI is necessary for further programming
##########################################################################   
    Pr = pic.Problem()
    Pr.set_objective('min', ((rho|BO) + (BO|rho))/2)
    Pr.add_list_of_constraints([pic.sum([P[x][a] for a in range(k)]) ==np.eye(d) \
 for x in range(nmeas[sys])],key="probabilities sum up to 1")
    Pr.add_list_of_constraints([P[x][a] >> 0 for a in range(k) \
 for x in range(nmeas[sys])],key="effects are PSD")
    S = Pr.solve(solver='mosek',verbose=0,tol=1e-7)
    return [[np.array(cvx.matrix(x.value)) for x in sett] for sett in P], S.value

def ranSDP(d):
    Mr = numpy.random.rand(d,d)
    Mi = numpy.random.rand(d,d)
    M = Mr + 1j*Mi
    return numpy.dot(M,dagger(M))

def randompovm(d=2,k=2):
    ranMs = [ranSDP(d) for i in range(k-1)]
    ranM = reduce(numpy.add,ranMs,numpy.array([[0]*d]*d))
    vas,ves = eigh(ranM)
    mva = vas[-1]
    res = [Mit/mva for Mit in ranMs] + [numpy.eye(d) - ranM/mva] # the last element in the POVM is not full rank, this is always possible for the extremal POVM, and we only need extremal POVMs for the maximal violation
    return res
