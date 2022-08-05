import numpy as np
import itertools as it
from bellpy.find.behavior import *

class Quantum_behavior(Behavior):
    def __init__(self, observables, state):
        self.observables = None
        self.dims = None
        self.set_observables(observables)

        self.state = None
        self.set_state(state)

        self.nsettings = [len(o)-1 for o in self.observables ]
        self.bs = Default_expectation_behavior_space(*self.nsettings)
        self.correlation_list = self.behavior_space.tuples

        self.table = [self.correlation(c) for c in self.correlation_list]

    @property
    def behavior_space(self):
        return self.bs
    
    def set_observables(self, observables):
        self.dims = [np.shape(x[0])[0] for x in observables]
        idds = [np.eye(d) for d in self.dims]
        self.observables = [ [idds[i]] + observables[i] for i in \
range(len(observables))]

    def set_state(self, state):
        d = np.prod(self.dims)
        if np.shape(state)==(d,d):
            self.state = state
        elif np.shape(state) == d:
            self.state = np.outer(state, state.conj())
        elif state==0:
            self.state = np.zeros([d,d])
        else:
            raise Exception("State has wrong dimension")
        
    def correlation(self, corr):
        obs = []
        if len(corr) == len(self.observables):
            for i in range(len(corr)):
                obs.append(self.observables[i][corr[i]])
        O = mkronf(obs)
        return np.einsum("ij,ji", O, self.state)
            

X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
    
       
def test1():
    q = QuantumBehavior([[np.random.random([2,2]) for i in range(3)] for j in range(3)]) 
    q.set_state(np.random.rand(8))
    print(q.correlation_list)

#test1()
def mkronf(systems):
    # taken from Felix Hubers qgeo.py
    """ fast tensor product
        fast tensor product of elements in list, takes repeatingly the tensor product (sp.kron)
        with the last element in systems until all subsystems tensored up.
        using einsum
        #skip empty subsystems
    args:       systems : array_like, ndarray or list
    returns:    ndarray
    """
    
    k = 0
    num = len(systems)
    dx = 1
    dy = 1
    params = []
    for s in systems:
        params.append(s)
        params.append([k, k+num])
        
        dx = dx * np.shape(s)[0]
        dy = dy * np.shape(s)[1]
        k+=1
        
    return np.einsum(*params).reshape(dx, dy)

