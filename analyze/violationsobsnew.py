import numpy as np
import scipy.linalg as la
import itertools as it
import bellpy.analyze.bellnp as bn

class Bellopt:
    def __init__(self, btab, dims = 2, min_iterations = 10, initsett=None, verbose=True, state=None, fix_state=False):
        self.nparties = len(np.shape(btab))
        self.btab = btab
        self.state = state
        self.dims = None
        self.n_iterations = min_iterations
        self.verbose = verbose
        self.fix_state = fix_state
        if initsett is None:
            self.get_init_settings(dims)
        else:
            self.settings = initsett
        self.violation = None

        self.seesaw()

    def seesaw(self, n=None):
        if n is None:
            n = self.n_iterations
        for i in range(n):
            if not self.fix_state:
                self.update_state()
            for p in range(self.nparties):
                self.update_settings(p)
        if self.verbose:
            print(self.violation)
        return 0

    def randomunitary(self, n):
        # math-ph/0609050.pdf
        M = (np.random.randn(n,n) + 1j*np.random.randn(n,n))/np.sqrt(2);
        q,r = np.linalg.qr(M);
        d = np.diagonal(r);
        d = d/np.absolute(d);
        q = np.multiply(q,d,q); # third argument is not a factor, it is output!
        return q


    def randomobs(self,dim):
        U = self.randomunitary(dim)
      # if dim==2:
      #     Z = np.array([[1, 0],[0, -1]],dtype=np.complex)
      # else:
        Z = np.diag(np.random.randint(0,2,dim)*2 - 1).astype(np.complex)
        return np.einsum('ij,jk,lk->il',U,Z,U.conj())


        
    def get_init_settings(self, dims):
        """
        Create random settings
        """
        if type(dims) is int:
            self.dims = [dims]*self.nparties
        else:
            self.dims = dims

        nsett = np.array(np.shape(self.btab)) - 1

        self.settings = [ [ self.randomobs(self.dims[i]) for j in \
                            range(nsett[i]) ] for i in range(self.nparties) ]
        
        return 0

    def update_state(self):
        Bo = bn.belloperator(self.btab, self.settings)
        lamda, V = np.linalg.eig(Bo)
        ind = np.argmin(lamda)
        self.state = V[:, ind]
        self.violation = lamda[ind]
        return 0

    def atilde(self,party, U):
        A_old = self.extended_settings()[party]
        Atil_old = [np.einsum('ij,jk,kl -> il', U.conjugate().transpose(), A, U) for A in A_old]
      # print('list lengths')
      # print(len(Atil_old))
      # print(len(X))
        atiloldarr = np.array(Atil_old)
        return atiloldarr

   #def behavior(self):
        

    def optimal_atil(self, x):
        D, U = la.eig(x)
        alph = -np.sign(D)
        return np.einsum('ij,j,jk', U, alph, U.conjugate().transpose())


    def update_settings_testing(self, party):
        U,li, Vst = self.schmidt(party) 
        X = self.x_operators(party, U, li, Vst)

        Bo = bn.belloperator(self.btab, self.settings)
        vold1 = np.einsum('i, ij, j', self.state.conjugate(), Bo, self.state)
        atiloldarr = self.atilde(party, U)
        xarr = np.array(X)
        vold2 = np.einsum('alk, akl', atiloldarr, xarr)
       #print('vls: ', v1, v2)
        tr1 = np.einsum('ijk,ikj->i', atiloldarr, xarr)

        Atil = [np.eye(self.dims[party])] + [ self.optimal_atil(x) for x in X[1:]]
        atilnewarr = np.array(Atil)
        tr2 = np.einsum('ijk,ikj->i', atilnewarr, xarr)
    #   print('tr2 < tr1', tr2 < tr1 + 1e-6)
    #   if not np.all(tr2<tr1):
    #       print('tr2 < tr1')
    #       print(tr2)
    #       print(tr1)
    #       print('.........')
    #       i = np.argmin(tr1 - tr2)
    #       print('i', i)
    #       print('x', xarr[i])
    #       print('aold', atiloldarr[i])
    #       print('anew', atilnewarr[i])
    #       print('vold', np.einsum('jk,kj', atiloldarr[i], xarr[i]))
    #       print('vnew', np.einsum('jk,kj', atilnewarr[i], xarr[i]))
    #       raise Exception("Optimization error")

            
        vnew2 = np.einsum('alk, akl', atilnewarr, xarr)
        A = [np.dot(U, np.dot(atil, U.conjugate().transpose())) for atil in Atil[1:]]
       #print('setting update')
       #print(self.settings)
        self.settings[party] = A
       #print(self.settings)
       #print('----------')
        Bo = bn.belloperator(self.btab, self.settings)
        vnew1 = np.einsum('i, ij, j', self.state.conjugate(), Bo, self.state)
        print('vl check:')
        print(vold1)
        print(vold2)
        print(vnew2)
        print(vnew1)
        print('-------------')
    
        return 0

    def update_settings(self, party):
        U,li, Vst = self.schmidt(party) 
        X = self.x_operators(party, U, li, Vst)
        Atil = [np.eye(self.dims[party])] + [ self.optimal_atil(x) for x in X[1:]]
        self.violation = np.einsum('alk, akl', np.array(Atil), np.array(X))
        A = [np.dot(U, np.dot(atil, U.conjugate().transpose())) for atil in Atil[1:]]
        self.settings[party] = A
        return 0
        

    def schmidt(self, party):
        """ 
        schmidt decomposition:
        party vs all the other partys
        """
        dprev = int(np.product(self.dims[0:party]))
        dafter = int(np.product(self.dims[party+1:]))
        d = int(self.dims[party])
        psi = np.reshape(self.state, [dprev, d, dafter ])
        psi = np.einsum("ijk -> jik", psi)
        psi = np.reshape(psi, [d, dprev*dafter]) 
        U, s, Vh = la.svd(psi)
#       print('shapes schmidt', np.shape(U), np.shape(s), np.shape(Vh))
#       Unull = la.null_space(U).transpose().conjugate()
#       Ufull = np.vstack((U,Unull))
#       dimkern = np.shape(Unull)[0]
#       print('dimkern', dimkern)
#       sfull = np.append(s, np.zeros(dimkern))
        Vcon = Vh.transpose()
#       Vconnull = la.null_space(Vcon).transpose().conjugate()
#       Vconfull = np.vstack((Vcon, Vconnull))
#       print('full shapes schmidt', np.shape(Ufull), np.shape(sfull), np.shape(Vconfull))
        return U, s, Vcon


    def extended_settings(self):
        """
        returns settings but adds identity as zeroth (trivial) setting
        """
        settlist = [ [np.eye(self.dims[i])] + self.settings[i] for i in range(len(self.settings))]
        return settlist


    def biseparate(self, party):
        """
        gives bipartite version of bell inequality with
        separation party vs. all other parties.
        Returns:
        Aops: Settings for party, including identity
        Bops: Settings for other parties, including identity
        w: weights of the Bell inequality  B = w_ij Aops[i] Bops[j]
        """

        settnums = np.array([len(s) for s in self.settings]) + 1
        dprev = np.product(settnums[0:party])
        dafter = np.product(settnums[party+1:])
        d = settnums[party]
        w = np.reshape(self.btab, [dprev, d, dafter ])
        w = np.einsum("ijk -> jik", w)
        w = np.reshape(w, [d, dprev*dafter]) 

       
        # subsume the settings of the other parties to one party
        othersettlist = self.settings.copy()
        othersettlist = [ [np.eye(self.dims[i])] + othersettlist[i] for i in range(len(othersettlist))]
        Aops = othersettlist.pop(party)
      # print('othersettlist', othersettlist)
        Bops = [bn.mkronf(i) for i in it.product(*othersettlist)]

        return Aops, Bops, w



    def check_biseparate(self, party=1):
        Aops, Bops, w = self.biseparate(party)
        Correlators = np.array([bn.mkronf(i) for i in it.product(Aops, Bops)])
        wflat = w.flatten()
        Belloperator = np.einsum('i, ijk-> jk', wflat, Correlators)
        Belloperator = reorder(Belloperator, self.dims,party)

        Bo = bn.belloperator(self.btab, self.settings)
        print(Bo)
        print(np.round(Bo -Belloperator, 5))
        return 0
        


    def x_operators(self, party, U, s, Vst):
        # Find the operator X as defined in the paper "more nonlocality with
        # less entanglement"
    
        Aops, Bops, w = self.biseparate(party)
        dimparty = self.dims[party]
        Btilops = [np.dot(Vst.conjugate().transpose(), np.dot(B, Vst))[0:dimparty,0:dimparty] for B in Bops]
        Btilops = np.array(Btilops)
        
        X = np.einsum("l,k,blk,ab -> akl", s, s, Btilops, w) 
        return list(X)

def violation_fix_state(B, dims, psi, iobs=None, verbose=False, numtol=1e-6, \
repetitions=17, initrep=7):
    Bviols = [ Bellopt(B, dims, initrep, initsett=iobs, verbose=False, state = psi,
fix_state=True) for i in range(15) ]
    vls = [vl.violation for vl in Bviols]
    k = np.argmin(vls)
    vk = Bviols[k]
    improv = 100
    i = 0
    while improv > numtol and i < repetitions - initrep:
        i += 1
        v1 = vk.violation
        vk.seesaw(1)
        improv = v1 - vk.violation
    return vk.violation, vk.settings


def violation(B, dims, repetitions=17, initvals=15, initrep=7,\
 numtol=1e-6,verbose=False, iobs=None):
    Bviols = [ Bellopt(B, dims, initrep, initsett=iobs, verbose=verbose) for i in range(initvals) ]
    vls = [vl.violation for vl in Bviols]
    k = np.argmin(vls)
    vk = Bviols[k]
    improv = 100
    i = 0
    while improv > numtol and i < repetitions - initrep:
        i += 1
        v1 = vk.violation
        vk.seesaw(1)
        improv = v1 - vk.violation
    return vk.violation, vk.state , vk.settings


def violation_after_settingsupdate(vk):
    Bo = bn.belloperator(vk.btab, vk.settings)
    lamda, V = np.linalg.eig(Bo)
    ind = np.argmin(lamda)
    return lamda[ind]


def violation_testing(B, dims, repetitions=17, initvals=15, initrep=7,\
 numtol=1e-6,verbose=False, iobs=None):
    Bviols = Bellopt(B, dims, 0, initsett=iobs, verbose=verbose) 
    vls = Bviols.violation
    print('check bisep', Bviols.check_biseparate())
    print("init violations:", vls)
    i = 0
    while i < 10:
        i += 1
        v1 = Bviols.violation
        Bviols.update_state()
        print('--------------------------------')
        print('f state upd', Bviols.violation)
        for p in range(Bviols.nparties):
            print('p', p)
            Bviols.update_settings(p)
            print(violation_after_settingsupdate(Bviols))
    return Bviols.violation, Bviols.state , Bviols.settings


def reorder(Op, dims, party):
    """
    Op = sum c_ijklmn |ijk><lmn| as matrix
    
    returns a version of Op, where system 0 will become system party 
    so, eg party = 3
    returns sum c_jkimnl |ijk><lmn|
    """
    inds = list(range(len(dims)))
    inds.pop(0)
    inds.insert(party,0)
    allinds = inds + [i + len(dims) for i in inds]
    allorginds = list(range(2*len(dims)))
    X = np.reshape(Op, dims + dims)
    X = np.einsum(X, allorginds, allinds)
    X = np.reshape(X, [np.prod(dims), np.prod(dims)])
    return X
