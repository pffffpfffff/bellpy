import numpy as np
import cdd
import itertools as it
import cvxopt as cvx
import mosek
import picos as pic
import scipy.linalg as la

class NoSignaling:
    """
    Intended for two-outcome meas +-1

    Given a list with the number of settings per party, 'settings',
    this class should provide the following:
    
    - A matrix to convert a ns behavior in probability formulation to obs
      formulation

    - write down equality and inequality constraints

    - (for small scenarios) solve the polytope, return vertices
        either in probabilities or in observables
    
    """
    def __init__(self, settings):
        self.parties = len(settings)
        self.settings = settings

        # for probabilities, effectively list, (out, inp)
        self.correlation_list = 0

        # for observables, effectively a list of inputs
        self.obs_corr_list = 0

        self.get_correlation_list()
        self.full_dim = len(self.correlation_list)
        self.obs_dim = len(self.obs_corr_list)

        self.to_obs = np.array([self.obs_dim, self.full_dim])
        self.calc_to_obs()

        self.NS = 0
        self.ns_constraints()

        self.inequalities = []
        self.get_inequalities()
        self.vertices = [] 

    def out_it(self, inp):
        outr = [ [-1, 1] if i!=0 else [1] for i in inp]
        for out in it.product(*outr):
            yield out

    def get_correlation_list(self):
        corr = []
        nsett = [list(range(s + 1)) for s in self.settings]
        for inp in it.product(*nsett):
            for out in self.out_it(inp):
                corr.append((out, inp))  
        self.obs_corr_list = list(it.product(*nsett))
       #print(nsett)
        self.correlation_list = corr
       #print(self.correlation_list) 
        return 0

    def to_obs_element(self, obscorr, probcorr):
        if obscorr == probcorr[1]:
            return np.prod(probcorr[0])
        else:
            return 0

    def calc_to_obs(self):
        T  = [[ self.to_obs_element(o, p) for p in self.correlation_list] for o in self.obs_corr_list]
        self.to_obs = np.array(T)
        return 0

    def inp_rel2marg(self, inp, marg):
        # input is related to some marg corr <=> marg can be obtained by summing
        # over some outcomes for input inp, input of marg has to be the same as
        # inp for all parties except the one, where no measurement takes place.
        # However, inp and marg[1] should not be identical
        if inp!=marg[1]:
            return all([ (inp[i] == marg[1][i] or marg[1][i]==0) for i in range(self.parties)])
        else:
            return False

    def out_rel2marg(self, out, marg, verb = False):
      # if verb:
      #     print([ (out[i] == marg[0][i] or marg[1][i]==0) for i in range(self.parties)])
        return all([ (out[i] == marg[0][i] or marg[1][i]==0) for i in range(self.parties)])

    def ns_element(self, corr, marg, inp):
        if corr == marg:
            return -1
        elif corr[1] == inp:
            return 1 if self.out_rel2marg(corr[0], marg) else 0
        else:
            return 0

    def test_ns_element(self):
        corr = ((-1,-1), (1,1))
        marg = ((1, -1),(0, 1)) 
        inp = (1,1)
        print("test..", self.ns_element(corr, marg,inp ))
        self.out_rel2marg(corr[1], marg, verb= True)
        
    def ns_constraints(self):
        self.NS = []
        margs = [ p for p in self.correlation_list if ( 0 in p[1])]
       #print('margs', margs)
        for m in margs:
            rel_inps = [ i for i in self.obs_corr_list if self.inp_rel2marg(i,m)]
           #print('marg', m, '\n rel_inps', rel_inps)
            for i in rel_inps:
                ns = [ self.ns_element(corr, m, i) for corr in self.correlation_list ]
                self.NS.append(ns)
       #self.NS = np.array(self.NS)
        return 0 

    def get_inequalities(self):
        mNS = [list(x) for x in -np.array(self.NS)] 
        idd = np.eye(self.full_dim - 1)
        zeros = np.transpose([np.zeros(self.full_dim - 1)])
        ones = np.transpose([np.ones(self.full_dim - 1)])
        pos = [ list(x) for x in np.hstack((zeros, idd)) ]
        smaller1 = [list(x) for x in np.hstack((ones, -idd))]
       #self.inequalities = np.array(pos + smaller1 + self.NS + mNS)
        self.inequalities = np.array(pos + smaller1)


    def get_vertices(self, G=None):
        # additional constraint Gx = 0
        if G is not None:
            G = np.dot(np.array(G), self.to_obs)
            G2 = np.vstack((G, np.array(self.NS)))
        else:
            G2 = np.array(self.NS)
        subsp = la.null_space(G2) # ker(G2) = cspace(subsp)
        ine = np.dot(self.inequalities, subsp)
        mat = cdd.Matrix(ine)
        mat.rep_type = cdd.RepType.INEQUALITY
        poly = cdd.Polyhedron(mat)
        ext = poly.get_generators()
        ext = np.dot(np.dot(np.array(ext), subsp.transpose()), self.to_obs.transpose())
        norm = np.diag(1/ext[:, 0])
        ext = np.dot(norm, ext)

       #print(ext)
        return ext

    def nonlocal_ns(self, G=None):
        # based on heuristic: nonlocal beh are shorter in obs notation
        # does not work because nothing forces the point to be extremal
        if G is not None:
            G = np.dot(np.array(G), self.to_obs)
            G2 = np.vstack((G, np.array(self.NS)))
        else:
            G2 = np.array(self.NS)
        subsp = la.null_space(G2) # ker(G2) = cspace(subsp)
        ssp = pic.Constant(cvx.matrix(subsp))
        print(np.shape(self.inequalities))
        print(np.shape(subsp))
        ine = pic.Constant(cvx.matrix(np.dot(self.inequalities, subsp)))
        P = pic.Problem()
        T = pic.Constant("T",  cvx.matrix(self.to_obs))
        x = pic.RealVariable("x", np.shape(subsp)[1])
        P.add_constraint(ine*x >= 0) 
        P.add_constraint((ssp*x)[0] == 1) 
        N = ssp.T*T.T*T*ssp
        P.set_objective("min", (x | N*x))
        P.solve(solver="mosek")
        return np.array((T*ssp*x).value)


    def ns_violation(self, b):
        """
        find maximal violation of bell inequality b, given as coefficients
        in observable notation
        i.e 

        min < x, b>
        s. t x is no signaling
        """
        ns = pic.Constant("NS", cvx.matrix(np.array(self.NS)))
        bb = pic.Constant('b', cvx.matrix(np.array(b)))
       #ine = pic.Constant('I', cvx.matrix(np.array(self.inequalities)))
        T = pic.Constant("T",  cvx.matrix(self.to_obs))
        P = pic.Problem()
        x = pic.RealVariable("x", self.full_dim)
        P.set_objective("min", (T*x | bb))
        P.add_constraint(ns*x == 0)
        P.add_constraint(x>=0)
        P.add_constraint(x[0]==1)
       #try:
        sol = P.solve(verbose=0, solver="mosek")
        xopt = np.array(x.value).flatten()
       #print('xopt', xopt)
        sopt = np.dot(np.dot(self.to_obs, xopt), b)
        return sopt, xopt

    def is_nosignaling(self, beh):
        """
        check whether behavior is inside the ns polytope,
        beh is given in obs space
        """

        b = pic.Constant('b', cvx.matrix(np.array(beh)))
        x = pic.RealVariable("x", self.full_dim)
        
        ns = pic.Constant("NS", cvx.matrix(np.array(self.NS)))
        T = pic.Constant("T",  cvx.matrix(self.to_obs))
        P = pic.Problem()
        P.add_constraint(ns*x == 0)
        P.add_constraint(x[0]==1)
        P.add_constraint(x>=0)
        P.add_constraint(T*x == b)
        P.set_objective("find")
        try:
            sol = P.solve(solver="mosek")
            return True
        except:
            return False

def test1():
    ns = NoSignaling([2,2])
    print(ns.correlation_list)
    print(ns.obs_corr_list)
  # print(np.array(ns.NS))
#   ns.test_ns_element()
#   ext = ns.get_vertices()
#   print(np.round(ext).astype(int))
    x = ns.nonlocal_ns()
    print(x)
#   b = [2, 0, 0, 0, -1, -1, 0, -1, 1]
#   print(b)
#   s = ns.ns_violation(b)[0]
#   print('s', s)
#   print(np.dot(ns.to_obs, s[1]))

#test1()
