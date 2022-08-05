import numpy as np
import cdd
from fractions import Fraction
import picos as pic
import cvxopt as cvx
from math import gcd


class Polyhedron:
    def __init__(self, vertices, numtype='fraction'):
        """
        first entry is 1 for vertices, 0 for rays
        """
        self.numtype = numtype
        self.vertices = vertices
        self.hrep = None

    def get_hrep(self):
        if self.hrep is None:
            vert = cdd.Matrix(self.vertices, number_type=self.numtype)
            vert.rep_type = cdd.RepType.GENERATOR
            poly = cdd.Polyhedron(vert)
            self.hrep = poly.get_inequalities()
        return self.hrep


    def facets_or_linearities(self, returnval = 0):
        facets = []
        linearities = []
       ##print(hrep) 
        hrep = self.get_hrep()
        for i in range(hrep.row_size):
            ieq = list(hrep[i]) 
            if i in hrep.lin_set:
                linearities.append(ieq)
            else:
               #bi = np.dot(ieq, T_tr)
                bi = ieq
                facets.append(bi)
        if returnval == 'facets':
            return facets
        elif returnval == 'linearities':
            return linearities
        else:
            return facets, linearities

    def facets(self):
        r = self.facets_or_linearities('facets')
        return r

    def linearities(self):
        return self.facets_or_linearities('linearities')

    def is_facet(self, f):
        dim = np.shape(self.vertices)[1]
        ret = False
        try: 
            satis = [ np.dot(v, f) >= 0 for v in self.vertices ]
            all_satis = all(satis)
            if not(all_satis):
                satis = [ np.dot(v, f) <= 0 for v in self.vertices ]
                all_satis = all(satis)
            if all_satis:
                satur = [v for v in self.vertices if np.dot(v, f) == 0]
                M = np.array(satur)
                R = np.linalg.matrix_rank(M)
                ret = R==dim-1
        except:
            raise Exception('Facet quality could not be verified')
        return ret

    def close_facet(self, f):
        dimp = np.size(self.vertices[0])
        P = pic.Problem()
        b = pic.RealVariable('b', dimp)
        V = pic.Constant(cvx.matrix(self.vertices))
        P.add_constraint(b[0] == 1)
       #P.add_constraint((1|b) - b[0] >=0.1)
        P.add_constraint(V * b >= 0)

        x = pic.Constant("x", f)
        P.set_objective("min", (x[1:] | b[1:]))
       #P.set_objective("find")
        P.solve(verbose=1, solver="mosek")
        bval = np.array(b.value).flatten()
       #print('bval', bval)
        bvalfrac = [Fraction(x) for x in bval]
        bvalfrac = [f.limit_denominator(1000) for f in bvalfrac]
        nv = self.expand_to_int([ Fraction(x) for x in bvalfrac])
        return nv

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
 
class Polytope(Polyhedron):
    def __init__(self, vertices, numtype='fraction'):
        self.numtype = numtype
        self.vertices = np.hstack((np.ones([np.shape(vertices)[0], 1]),
            vertices))
        self.hrep = None

class Cone(Polyhedron):
    def __init__(self, vertices, numtype='fraction'):
        self.numtype = numtype
        self.vertices = np.hstack((np.zeros([np.shape(vertices)[0], 1]),
            vertices))
        self.hrep = None





import itertools as it
def test():
    verts = [ np.array([1] + list(i)) for i in it.product([-2,2], repeat = 3)]
    P = Polytope(verts)
    print(P.facets())

#test()
#       dim = np.shape(self.vertices)[1]
#       ret = False
#       norm = lambda v: np.sqrt(np.dot(v,v))
#       # devide vertices into four subsets:
#       # points left of hyperplane f
#       # points right of hyperplane f
#       # points on hyperplane f
#       # points close to hyperplane f
#       onleft = []
#       onright = []
#       deadon = []
#       close = []
#       for v in self.vertices:
#           dotprod = np.dot(v, f)
#           normv = norm(v)
#           if dotprod > 1e-3 * normv:
#               onright.append(v)
#           elif dotprod < -1e-3 * normv:
#               onleft.append(v)
#           elif dotprod == 0:
#               deadon.append(v)
#           else:
#               close.append(v)
#       try: 
#           satis = [ np.dot(v, f) >= 0 for v in self.vertices ]
#           all_satis = all(satis)
#           if not(all_satis):
#               satis = [ np.dot(v, f) <= 0 for v in self.vertices ]
#               all_satis = all(satis)
#           if all_satis:
#               satur = [v for v in self.vertices if np.dot(v, f) == 0]
#               M = np.array(satur)
#               R = np.linalg.matrix_rank(M)
#               ret = R==dim-1
#       except:
#           raise Exception('Facet quality could not be verified')
#       return ret


