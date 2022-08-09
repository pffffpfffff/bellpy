from sympy.combinatorics.named_groups import SymmetricGroup
import sympy as sym
from sympy.combinatorics import Permutation, PermutationGroup
#from setting import Setting
from abc import ABC, abstractmethod
from bellpy.find.behavior import *
from bellpy.find.probbehaviorspace import *
from bellpy.find.partysym import Permutation_from_image, PermutationMat
from bellpy.find.setting import Setting
from copy import deepcopy

class Relabeling(ABC):
    # requires member
    #   self.matrix

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, M):
        self._matrix = M

    def call_on_vector(self, v):
        if not isinstance(v, Behavior) and not isinstance(v, Bell_inequality):
            raise TypeError("Relabeling should be called on Behavior or Bell_inequality")
        if self.matrix is None:
            self.compute_matrix(v.behavior_space)
        vtab = np.dot(self.matrix, v.table) 
        if isinstance(v, Behavior):
            return Behavior(v.behavior_space, vtab)
        if isinstance(v, Bell_inequality):
            return Bell_inequality(v.behavior_space, vtab)
        raise Exception("Unknown Error when calling Relabeling")

    @abstractmethod
    def __call__(self, x):
        pass

    def compute_matrix(self, bs: Behavior_space):
        if isinstance(bs, NSpb_space):
            fbs = bs.fullspace
        elif isinstance(bs, Probability_behavior_space):
            fbs = bs
        else:
            raise Exception("Error in Setting_permutation: behavior space should either be NSpb_space or Probability_behavior_space")
        relevents = []
        for e in fbs.events:
            relsettings = [ self.__call__(s) for s in e ]
            relevents.append(Joint_measurement(relsettings))
        rellabels = [x.__str__() for x in self.events]
        self.matrix = PermutationMat(Permutation_from_image(fbs.labels, rellabels))
        return 0


class Outcome_permutation(Relabeling):

    def __init__(self, *args, setting = None, size = None, **kwargs):
        self.permutation = Permutation(*args, size = size, **kwargs)
        if not isinstance(setting, Setting):
            raise Exception("Provide adequate setting with setting keyword for Outcome_permutation")
        self.setting = setting

    def __call__(self, x):
        if isinstance(x, Setting):
            new_setting = copy.copy(x)
            if x.party == self.setting.party and x.label == self.setting.label:
                new_setting.value = self.permutation(new_setting.value)
            return newsetting
        return self.call_on_vector(self, x)

    

class Setting_permutation(Relabeling):

    def __init__(self, *args, party = None, size = None, **kwargs):
        self.permutation = Permutation(*args, size = size, **kwargs)
        if not isinstance(party, int):
            raise Exception("Provide adequate setting with party keyword for Setting_permutation")
        self.party = party

    def __call__(self, x):
        if isinstance(x, Setting):
            new_setting = copy.copy(x)
            if x.party == self.party:
                new_setting.label = self.permutation(new_setting.label)
            return newsetting
        return self.call_on_vector(self, x)

class Party_permutation(Relabeling):
    def __init__(self, size = size, **kwargs):
        self.permutation = Permutation(*args, size = size, **kwargs)

    def __call__(self, x):
        if isinstance(x, Setting):
            new_setting = copy.copy(x)
            new_setting.party = self.permutation(new_setting.party)
            return newsetting
        return self.call_on_vector(self, x)



