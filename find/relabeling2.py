from sympy.combinatorics.named_groups import SymmetricGroup
import sympy as sym
from sympy.combinatorics import Permutation, PermutationGroup
from abc import ABC, abstractmethod
from copy import deepcopy
import warnings
from functools import reduce
from numpy.linalg import multi_dot
from bellpy.find.behavior import *
import bellpy.find.probbehaviorspace as pbs
from bellpy.find.partysym import Permutation_from_image, PermutationMat
from bellpy.find.setting import Setting
from functools import lru_cache, reduce
import pdb

class ARelabeling(ABC):

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, M):
        self._matrix = M

    @property
    def behavior_space(self):
        return self._bs

    @behavior_space.setter
    def behavior_space(self, bs: Behavior_space):
        self._bs = bs

    def call_on_vector(self, v):
        if not isinstance(v, Behavior) and not isinstance(v, Bell_inequality):
            raise TypeError("Relabeling should be called on Behavior or Bell_inequality")
        if self.matrix is None:
            self.compute_matrix()
        if isinstance(v, Behavior):
            vtab = np.dot(self.matrix, v.table)
        if isinstance(v, Bell_inequality):
            vtab = np.dot(self.matrix.transpose(), v.table)
        return type(v)(v.behavior_space, vtab)
        
    @abstractmethod
    def __call__(self, x):
        pass

    def compute_matrix(self):

        relevents = []
        for e in self.behavior_space.events:
           # print([s for s in e])
            relsettings = [ self.__call__(s) for s in e.settings ]
            relevents.append(Joint_measurement(relsettings))
        rellabels = [x.__str__() for x in relevents]

        if isinstance(self.behavior_space, pbs.NSpb_space):
            inv = self.behavior_space.conversion_matrix_to_standard_basis(rellabels)
        elif isinstance(self.behavior_space, pbs.Probability_behavior_space): 
            inv = PermutationMat(~Permutation_from_image(self.behavior_space.labels, rellabels))
        else:
            raise Exception("invalid behavior space, should be either NSpb_space or Probability_behavior_space")
        
        self.matrix = inv
           
        return 0

class Relabeling(ARelabeling):

    def __init__(self, *relabelings, matrix = None):
        self.relabelings = relabelings
        self.behavior_space = reduce(lambda x, y: x or y, [r.behavior_space for r in self.relabelings])
#       for r in self.relabelings:
#           r.compute_matrix()
        if not matrix:
            self._matrix = multi_dot([r.matrix for r in self.relabelings])
        else:
            self._matrix = matrix

    def __mul__(self, other):
        return Relabeling(np.dot(self.matrix, other.matrix))

    def __call__(self, x):
        return self.call_on_vector(x) 


class Outcome_permutation(Relabeling):

    def __init__(self, *args, permutation = None, setting = None, behavior_space
= None, size = None, **kwargs):
        self.behavior_space = behavior_space
        if permutation is None:
            self.permutation = Permutation(*args, size = size, **kwargs)
        else:
            self.permutation = permutation
            if size or args or kwargs:
                warnings.warn("if a permutation is passed to Relabeling, args, kwargs and size will be ignored")
                
        if not isinstance(setting, Setting):
            raise Exception("Provide adequate setting with setting keyword for Outcome_permutation")
        self.setting = setting
        self.compute_matrix()

    def __call__(self, x):
        if isinstance(x, Setting):
            new_setting = copy.copy(x)
            if x.party == self.setting.party and x.label == self.setting.label:
                new_setting.value = self.permutation(new_setting.value)
            return new_setting
        return self.call_on_vector(x)

    

class Setting_permutation(Relabeling):

    def __init__(self, *args, permutation = None, behavior_space = None, party = None, size = None, **kwargs):
        self.behavior_space = behavior_space
        if permutation is None:
            self.permutation = Permutation(*args, size = size, **kwargs)
        else:
            self.permutation = permutation
            if size or args or kwargs:
                warnings.warn("if a permutation is passed to Relabeling, args, kwargs and size will be ignored")

        if not isinstance(party, int):
            raise Exception("Provide adequate setting with party keyword for Setting_permutation")
        self.party = party
        self.compute_matrix()

    def __call__(self, x):
        if isinstance(x, Setting):
            new_setting = copy.copy(x)
            if x.party == self.party:
                new_setting.label = self.permutation(new_setting.label)
            return new_setting
        return self.call_on_vector(self, x)

class Party_permutation(Relabeling):
    def __init__(self, *args, permutation = None, behavior_space = None, size = None, **kwargs):
        self.behavior_space = behavior_space
        if permutation is None:
            self.permutation = Permutation(*args, size = size, **kwargs)
        else:
            self.permutation = permutation
            if size or args or kwargs:
                warnings.warn("if a permutation is passed to Relabeling, args, kwargs and size will be ignored")

        self.compute_matrix()

    def __call__(self, x):
        if isinstance(x, Setting):
            new_setting = copy.copy(x)
            new_setting.party = self.permutation(new_setting.party)
            return new_setting
        return self.call_on_vector(self, x)

def le_vectors(v1: np.array, v2: np.array):
    """ 
    Use lexsort to order vectors, lexsort sorts columns
    """
    V = np.array([v1, v2]).transpose()
    try:
        le = (np.lexsort(V)[0] == 0)
    except:
        print('V', V)
        raise Exception
    return le

class AGroup(ABC):
    @abstractmethod
    def __iter__(self):
        pass

####@lru_cache
    def normal_forms(self, objects, orderfct = lambda x, y: le_vectors(x.table, y.table)):
        dct = {o: o for o in objects}
        for g in self:
            for o, oo in dct.items():
                if orderfct(g(o), oo):
                    dct[o] = g(o)
        return dct

    def representants(self, objects, orderfct = lambda x, y: le_vectors(x.table, y.table)):
        nf = self.normal_forms(objects, orderfct)
        return list(set(nf.values()))


class Relabelings_group(AGroup):
    def __init__(self, group: PermutationGroup, behavior_space: Behavior_space, action, party = None, setting = None):
        self.group = group
        self.behavior_space = behavior_space
        self.acts_on = action
        if action == "parties":
            self.action = lambda perm: Party_permutation(permutation =
perm, behavior_space = self.behavior_space)
        elif action == "settings":
            if not isinstance(party, int):
                raise Exception("Provide party on which settings should be permuted")
            self.action = lambda p: Setting_permutation(permutation = p, party =
party, behavior_space = self.behavior_space)
        elif action == "outcomes":
            if not isinstance(setting, Setting):
                raise Exception("Provide setting on which outcomes should be permuted")
            self.action = lambda p: Outcome_permutation(permutation = p, setting
= setting, behavior_space = self.behavior_space)
        else:
            raise Exception('action of Relabelings_group is either "parties", "settings", or "outcomes"') 

        self.cache = dict({})

    def __iter__(self):
        return RG_iter(self)


class RG_iter:
    def __init__(self, rg: Relabelings_group):
        self.rg = rg
        self.elements = rg.group.elements
        self.action = rg.action
        self.iter = iter(self.elements)

    def __next__(self):
        g = next(self.iter)
        if g not in self.rg.cache:
            self.rg.cache[g] = self.action(g)
        return self.rg.cache[g]


class Product_group(AGroup):
    def __init__(self, *groups):
        self.groups = groups

    def __iter__(self):
        return Prgr_iter(self)

class Prgr_iter:
    def __init__(self, pg: Product_group):
        self.iter = it.product(*pg.groups)
    def __next__(self):
        g = next(self.iter)
        return Relabeling(*g)
 
Outcomes_group = lambda behsp, setting: \
Relabelings_group(SymmetricGroup(len(setting.outcomes)), behsp, "outcomes", setting = setting)

Settings_group = lambda group, behsp, party: Relabelings_group(group, behsp, "settings", party = party)

Parties_group = lambda group, behsp: Relabelings_group(group, behsp, "parties")        
