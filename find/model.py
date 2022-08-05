import abc
import itertools as it
import pandas as pd
import numpy as np
from bellpy.find.assignments import *
from bellpy.find.behavior import *
from bellpy.find.polytope import *

class Model(abc.ABC):

    @property
    @abc.abstractmethod
    def behavior_space(self) -> Behavior_space:
        pass


    @abc.abstractmethod
    def __iter__(self):
        pass


    def data_frame(self):
        lst = []
        for beh in self:
            lst.append(beh.table)
      # lst = list(set(lst))
      # print('beh to array')
      # print(type(beh.to_array()))
      # print(beh.to_array())
        labels = self.behavior_space.labels
        return pd.DataFrame(lst, columns = labels, dtype=int)


    def bell_inequalities(self):
        P = Polyhedron(self.data_frame().to_numpy())
        fac = P.facets()
       #print('facets', fac)
        return self.behavior_space.facets_to_bell_inequalities(fac)

    def __str__(self):
        return self.data_frame().__str__()



class Unrestricted_model(Model):
    def __init__(self, behavior_space: Behavior_space):
        self.bs = behavior_space
        self.sc = self.bs.scenario
        
    @property
    def assignment_model(self):
        return Unrestricted_assignments(self.sc, mode=self.bs.mode)

    @property
    def behavior_space(self):
        return self.bs

    def __iter__(self):
        return Model_iterator(self)

class Model_iterator:
    def __init__(self, model):
        self.model = model
        self.bs = self.model.behavior_space
        self.iter = iter(self.model.assignment_model)

    def __next__(self):
        return self.bs.behavior_from_assignments(next(self.iter))

class Local_deterministic_model(Model):

    def __init__(self, bs: Behavior_space, partition = None):
        self.bs = bs
        self.sc = self.bs.scenario
        self.partition = partition
    
    @property
    def assignment_model(self):
        lda = Local_deterministic_assignments(self.sc, self.partition, mode=self.bs.mode)
        return lda

    @property
    def behavior_space(self):
        return self.bs

    def __iter__(self):
        return Model_iterator(self)

class Hybrid_model(Model):
    def __init__(self, *ldms):
        self.ldms = ldms # Local_deterministic_models

    @property
    def behavior_space(self):
        return self.ldms[0].behavior_space

    def __iter__(self):
        return it.chain(*self.ldms)


