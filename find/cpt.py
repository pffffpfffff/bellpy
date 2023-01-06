import abc
import copy
import itertools as it
import mosek
import picos as pic
import cvxopt as cvx
import numpy as np
import sympy as sym
import pickle
import os.path
from typing import List

from fractions import Fraction
from math import gcd
from scipy.linalg import null_space, orth

from bellpy.find.model import *


class Cone_projection_technique:
    def __init__(self, model, G=None, true_facets = False, vertices = None, name
            = 'facets'):
       #vertices = model.data_frame().to_numpy()
        self.model = model
        self.name = name
        self.behavior_space = self.model.behavior_space
        if vertices is None:
            self.vertices = self.model.data_frame().to_numpy()
        else:
            self.vertices = vertices

        self.vertices = np.array(self.vertices).astype(int)
        if G is None:
            self.G = None
        else:
            self.G = np.array(G).astype(int) # affine constraints

        self.truefacets = true_facets
        self.pv = None
        self.T = None
        self.linearities = None

        """
        add search space, like G defines a subspace in which facets are
        searched, however treat it differently:

        - For constraints in G
        facet normal vectors should satisfy constraint exactly

        - For constraints that define search space
        find facets search space and using them find facets of the original
        polytope (or the one projected into ker(G))

        Define search space by matrix S

        G should have integer coefficients
        S can be float, will be approximated by Fraction
        """
        self.S = None
        self.pv2 = None
        self.T2 = None

    def set_model(self, model: Model):
        self.model = model

    def add_extended_behaviors(self, extbeh):
        ext = np.round(extbeh.data_frame().to_numpy()).astype(int)
        self.add_constraints(ext)
       #print('G', self.G, np.shape(self.G))
        return 0

    def add_constraints(self, const):
        if self.G is None:
            self.G = const.astype(int)
        else:
            self.G = np.round(np.vstack((self.G, const))).astype(int)
       #print('G', self.G, np.shape(self.G), np.linalg.matrix_rank(self.G))
        return 0

    def add_party_symmetry(self):
        self.add_constraints(self.behavior_space.party_symmetry())
        return 0

    def add_only_full_body_correlations(self):
        try:
            self.add_constraints(self.behavior_space.full_body_correlations())
        except:
            raise Exception('failed to add condition <full_body_correlations>,\
                    implemented for this behavior space?')
        return 0

    def set_search_space(self, S):
       # self.S = [ [ Fraction(x) for x in y ] for y in S ] 
       self.S = S
       return 0


    def remove_close_vertices(self, Vl: List[np.array]):
        V = copy.copy(Vl)
        def distances(i, L):
            # L list of vertices
            d = []
            for k in range(i):
                v = L[i] - L[k]
                dist = np.sqrt(np.dot(v,v))
                d.append(dist)
            return d

        i = 1
        while True:
            mindist = min(distances(i, V))
            if mindist < 1e-3:
                V.pop(i)
            else:
                i += 1
            if i >= len(V):
                break

        return V


    def proj_verts(self):
        # returns "projection" of vertices into ker(G)
        G  = self.G

        if (G is None):
            T = np.eye(np.shape(self.vertices)[1])
        else:
            T = sym.Matrix(G).nullspace()
            T = np.array(T)
            T = T.transpose()[0]
           #T = np.round(T)
            T = T.astype(int)
           #print('T', T, type(T))

        self.T = T
        print('num vertices', np.shape(self.vertices))
        V = np.dot(self.vertices,T)#.astype(float)
        V = [tuple(x) for x in V]
        V = set(V)
        V = [list(x) for x in V]
        print('unique proj verts', np.shape(V))
        self.pv = np.round(np.array(V)).astype(int)
        return 0


    def proj_verts_search_space(self):
        # returns "projection" of vertices into intersection (ker(G) , ker(S))

        if not (self.S is None):
            print('projection into search space')
           #S = [[ Fraction(x) for x in y ] for y in self.S]
            S = self.S
            
            if (self.G is None):
                G = S
            else:
               # G = [ [ Fraction(x) for x in y] for y in self.G] + S
                G = np.vstack((self.G, S))

            print('condition matrix created, computing nullspace')
            print('G', G)
            T = null_space(G)
            T = orth(T)
           #T = sym.Matrix(G).nullspace()
           #T = np.array(T).astype(float)
           #T = np.round(T).astype(int)
            T = np.array(T)
           #try:
           #    T = T.transpose()[0]
           #except:
           #    raise Exception("Transposition failed, T: ", T)

            print('T', T)
            self.T2 = T
           #print('num vertices', np.shape(self.vertices))
            V = np.dot(self.vertices,T)#.astype(float)
          # print('verts', self.vertices)
          # print('V', V)
          # print('V', type(V))
            V = [tuple(x) for x in V]
            V = set(V)
            if not(self.S is None):
                V = self.remove_close_vertices([np.array(v) for v in V])
            else:
                V = [list(x) for x in V]
           #print('unique proj verts', np.shape(V))
           #self.pv = np.round(np.array(V)).astype(int)
            self.pv2 = np.array(V)
        return 0

    @staticmethod
    def expand_to_int(lof):
        loi = [x.denominator for x in lof]
        lcm = loi[0]
        for i in loi[1:]:
            lcm = np.round(lcm).astype(int)
            i = np.round(i).astype(int)
           #print('gcd error', lcm, type(round(lcm),i, type(i))
            lcm = lcm*i/gcd(lcm, i)
        return [np.round(x*lcm) for x in lof]
 

    def proj_to_unproj(self, pf, T):
        T_tr = T.transpose()
        process_pf = lambda i: np.array(self.expand_to_int(i)).astype(int)
        return np.dot(process_pf(pf), T_tr)

    def bell_inequalities(self, interactive = False) -> Bell_inequality_collection:
        self.proj_verts()
        self.proj_verts_search_space()
        if self.pv2 is None:
            pv = self.pv
            T = self.T
        else:
            pv = self.pv2
            T = self.T2

        facets = []

        proj_cone = Cone(pv)
        P = Polyhedron(self.vertices)

        def calc():
            proj_facets = proj_cone.facets()
            for pf in proj_facets:
                pf = pf[1:]
                f = self.proj_to_unproj(pf, T)
                if self.pv2:
                    f = self.project_to_kerG(P.close_facet(f))
                if self.truefacets:
                    if P.is_facet(f) and self.check_condition(f):
                        facets.append(f)
                elif self.check_condition(f):
                    facets.append(f)
                else:
                    raise Exception('Conditions not met in cpt')

        if interactive:
            cont = input('Do you want to continue? [y/n]')
            if cont=='y':
                calc()
            else:
                raise Exception('Program terminated by user')
        else:
            calc()

        return self.behavior_space.facets_to_bell_inequalities(facets)

    def get_random_direction(self):

        if not (self.S is None):
            S = np.array(self.S)#.astype(float)
            dim = np.shape(S)[0] 
            direc = 2*(np.random.randn(dim)-1)
            direc = np.dot(direc, S)
            direc[0] = 1

        else:
            dim = np.size(self.vertices[0])
            direc = 2*(np.random.randn(dim)-1)
            direc[0] = 1

        return direc


    def random_bell_inequalities(self, n=1):
        dimp = np.size(self.vertices[0])
        P = pic.Problem()
        b = pic.RealVariable('b', dimp)
        V = pic.Constant(cvx.matrix(self.vertices))
        P.add_constraint(b[0] == 1)
       #P.add_constraint((1|b) - b[0] >=0.1)
        P.add_constraint(V * b >= 0)
        if not( self.G is None):
            G = pic.Constant(cvx.matrix(self.G))
            P.add_constraint(G * b == 0)

        bell_ines = []
        for i in range(n):
            direction = self.get_random_direction()
            x = pic.Constant("x", direction)
            P.set_objective("min", (x[1:] | b[1:]))
           #P.set_objective("find")
            P.solve(verbose=1, solver="mosek")
            bval = np.array(b.value).flatten()
           #print('bval', bval)
            bvalfrac = [Fraction(x) for x in bval]
            bvalfrac = [f.limit_denominator(1000) for f in bvalfrac]
            nv = self.expand_to_int([ Fraction(x) for x in bvalfrac])
            bi = self.behavior_space.facet_to_bell_inequality(nv)
            bell_ines.append(bi)
            print("{} out of {}".format(i+1, n))
       #nv = bval
        return bell_ines



    def bell_inequality_for_behavior(self, behavior):
        dimp = np.size(self.vertices[0])
        P = pic.Problem()
        b = pic.RealVariable('b', dimp)
        V = pic.Constant(cvx.matrix(self.vertices))
        G = pic.Constant(cvx.matrix(self.G))
        P.add_constraint(b[0] == 1)
       #P.add_constraint((1|b) - b[0] >=0.1)
        P.add_constraint(V * b >= 0)
        P.add_constraint(G * b == 0)

        direction = behavior.table
        x = pic.Constant("x", direction)
        P.set_objective("min", (x[1:] | b[1:]))
       #P.set_objective("find")
        P.solve(verbose=1, solver="mosek")
        bval = np.array(b.value).flatten()
       #print('bval', bval)
        bvalfrac = [Fraction(x) for x in bval]
        bvalfrac = [f.limit_denominator(1000) for f in bvalfrac]
        nv = self.expand_to_int([ Fraction(x) for x in bvalfrac])
        bell_ine = self.behavior_space.facet_to_bell_inequality(nv)
       #nv = bval
        return bell_ine



    def linearities(self):
        proj_cone = Cone(self.pv)
        linearities = [ self.proj_to_unproj(l) for l in proj_cone.linearities()]
        return linearities

    def check_condition(self, f):
        ret = False
        try:
            if self.G is None:
                ret = True
            else:
                ret = np.all(np.dot(self.G,f) == 0)
        except:
            raise Exception('Conditions could not be checked')
        return ret


    def write_ext_rational(self):
        print('Writing to ext file')
        if not( self.pv2 is None):
            C = self.pv2
            C = [[ Fraction(x).limit_denominator(10000) for x in c ] for c in C ]
        else:
            C = self.pv
       #C = list(set([tuple(x) for x in C]))
       #C = np.array(C)

       #print(C)
       ##print(type(C), 'type of C')
        text = """
V-representation
begin
{} {} rational
""".format(np.shape(C)[0],np.shape(C)[1]+1)
        for i in range(np.shape(C)[0]):
            text = text + " 0"
            if i%100 == 0:
                print('i: ', i)
            for j in range(0,np.shape(C)[1]):
                text = text + " {}".format(C[i][j])

            text = text + """
"""

        text = text + """
end"""
        f = open(self.name+".ext","w")
        f.write(text)
        f.close()
        return 0
    
    def save(self):
        try:  
            f = open(self.name+".cpt", 'wb')
            pickle.dump(self, f)
        except:
            print('could not save ' + self.name + '.cpt')
        return 0

    def project_to_kerG(self, f):
        return np.dot(self.T, np.dot(self.T.transpose(), f))

    def facets(self):
        # Returns normal vectors b of facets of the cone that satisfy G b = 0
        # In order to be successful, projected cone must already be solved
        Facets = []
        textfile = self.name + '.ine'
       ##print('T', self.T)
        if not (self.T2 is None):
            T = self.T2
            print('T2', T)
        else:
            T = self.T
        T_tr = T.transpose()
        P = Polyhedron(self.vertices, 'integer')
      
        with open(textfile, 'r') as inefile:
            i = 0
            alllines = inefile.readlines()
            while True:
                line = alllines[i]
                if line[0:5] == 'H-rep':
                    i += 1
                    break
                i += 1
            while True:
               line = alllines[i]
               if line.lstrip()[0] == '0':
                   break
               else:
                   i+=1

            for line in alllines[i:-1]:
                if line.lstrip()[0]!='0':
                    break
                f = line.split()[1:]
               #f = [float(i) for i in l[1:]]
                print('f proj', f)
                f = [Fraction(i) for i in f]
                f = np.dot(T, f)
                f = [x/f[0] for x in f]
                f = [Fraction(i) for i in f]
                f = [x.limit_denominator(100) for x in f]
                f = np.array(self.expand_to_int(f)).astype(int)
               #print('f', f)
                if not( self.T2 is None):
                    f = self.project_to_kerG(P.close_facet(f))
                if self.truefacets:
                    if P.is_facet(f) and self.check_condition(f):
                        Facets.append(f)
                elif self.check_condition(f):
                    Facets.append(f)
                else:
                    raise Exception('Conditions not met in cpt')
                    
        return Facets

    def run(self, cleanup = False):
        # if name.ine exists --> facets
        # if does not exist  --> create it
        if cleanup:
            self.delete_files()

        ine = os.path.isfile(self.name+".ine")
        ext = os.path.isfile(self.name+".ext")
        if ine:
            F = self.facets()
           #print(F)
            return self.behavior_space.facets_to_bell_inequalities(F)

        elif ext:
            print("Please convert the existing .ext file into .ine format")
          #os.system('for x in $(find . -iname "cone150720*.ext"); do lcdd_gmp
          #$x ${x%.ext}.ine; done;')
        else:
           # neither of the files exist
            self.proj_verts()
            self.proj_verts_search_space()
            self.write_ext_rational()
            self.save()
        return 0

    def delete_files(self):
        inefile = self.name+".ine" 
        extfile = self.name+".ext" 
        cptfile = self.name+".cpt" 

        if os.path.isfile(inefile):
            os.remove("./"+ inefile)
        if os.path.isfile(extfile):
            os.remove("./"+ extfile)
        if os.path.isfile(cptfile):
            os.remove("./"+ cptfile)
        return 0




def load_cpt(name):
    with open(name+'.cpt','rb') as f:
        return pickle.load(f)           



class Extension_map(abc.ABC):

    @property
    @abc.abstractmethod
    def domain(self):
        # return the input behavior space
        pass


    @property
    @abc.abstractmethod
    def codomain(self):
        # return the target behavior space
        pass
     
    @abc.abstractmethod
    def __call__(self,beh):
        """
        input: behavior in small behavior space
        output: behavior in large behavior space
        """
        pass

class Extended_behaviors(Model):
    def __init__(self, model, bell_inequality: Bell_inequality, em: Extension_map):
        self.model = model
        self.bi = bell_inequality
        self.em = em

    def __iter__(self):
        return Extended_behavior_iterator(self)

    @property
    def behavior_space(self):
        return self.em.codomain


class Extended_behavior_iterator:
    def __init__(self, eb):
        self.model = eb.model
        self.bi = eb.bi
        self.iter = iter(self.model)
        self.extension_map = eb.em

    def __next__(self):
        b = next(self.iter)
        while self.bi(b) != 0:
            b = next(self.iter)
           #print(f"Ext beh: type of b {type(b)}")
        return self.extension_map(b)

"""
Extension map for canonic expectation value behavior
"""

class Canonic_single_extension_map(Extension_map):

    def __init__(self, behsp: Behavior_space, party: int, value=0, sign=1):
        self.party = party
        self.value = value
        self.sign = sign
        self.behavior_space = behsp
        self.codom = None
        self.codomain
        """
        add one more setting on party 'party'.
        labels of settings are integers 0, 1, ...
        new setting has largest label, say M
        value is integer 0, 1, ..., M-1,
        new setting behaves like the one of the same party with label value

        Example:
        party = 0, value = 1, sign = -1
        Then A_M = -A_1

        """

    @property
    def domain(self):
        return self.behavior_space

    @property
    def codomain(self):
        if self.codom is None:
            nsettings = self.behavior_space.nsettings
            newnsettings = list(nsettings)
            if self.party == len(nsettings):
                newnsettings.append(0)
                # trivial setting is not counted and is always added automatically
            if not self.party in range(len(newnsettings)):
                raise Exception('Do not skip party labels')
            if not self.value in range(newnsettings[self.party]+1):
                raise Exception('<value> out of bounds')
            newnsettings[self.party] += 1
            self.codom = Default_expectation_behavior_space(*newnsettings)
        return self.codom

        


       #scen = self.behavior_space.scenario
       #partysett = scen.subscenario([self.party]).nsettings[0]
       #newsetting = Setting(self.party, partysett + 1, [-1, 1])
       #settings = scen.settings + newsetting
       #return Expectation_behavior_space(Scenario(settings))

    def __call__(self, behavior):
        table = behavior.to_canonic_array()
      # print('table', table)
        nparties = self.domain.scenario.nparties
        if self.party>=nparties:

            # setting belongs to new party
            # in this case, value can be ignored, has to be 0
            newtable = np.array([table, sign*table])
            newtable = np.einsum(newassignments, [0, Ellipsis], [Ellipsis, 0])

        else:
            # create
            newnsettings = self.codomain.scenario.nsettings
          # print('newnsettings', newnsettings)
          # print('oldnsettings', self.domain.scenario.nsettings)
        
            newtable = np.zeros([s for s in newnsettings])
            # sort affected party as first party
            oldorder = list(range(nparties))
            neworder = [self.party]+ [ i for i in range(nparties) if i != self.party]

            newtable = np.einsum(newtable, oldorder, neworder)
            oldtable = np.einsum(table, oldorder, neworder)
            newtable[0:self.domain.scenario.nsettings[self.party]] = oldtable
            newtable[-1] = self.sign*oldtable[self.value]
            newtable = np.einsum(newtable, neworder, oldorder)
        return Behavior(self.codomain, newtable.flatten())



class Canonic_extension_map(Extension_map):

    def __init__(self, behsp: Default_expectation_behavior_space, *pvs):
        self.behavior_space = behsp
        self.pvs = pvs
        self.single_extension_maps = self.compute_sems()

    def compute_sems(self):
        dom = self.behavior_space
        emaps = []
        for ext in self.pvs:
            emap = Canonic_single_extension_map(dom, *ext) 
            emaps.append(emap)
            dom = emap.codomain
        return emaps

    def __call__(self, beh):
        newbeh = beh
        for emap in self.single_extension_maps:
            newbeh = emap(newbeh)
        return newbeh

    @property
    def domain(self):
        return self.behavior_space

    @property
    def codomain(self):
        return self.single_extension_maps[-1].codomain



