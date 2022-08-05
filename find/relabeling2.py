from sympy.combinatorics.named_groups import SymmetricGroup
import sympy as sym
from sympy.combinatorics import Permutation
from bellpy.find.setting import Setting
#   class Outcome_group:
#       def __init__(self, sett: Setting):
#           self.sett = sett
#           self.group = SymmetricGroup(len(self.sett.outcomes))

class listperm(Permutation):
    def __call__(self, l):
        inds = list(range(len(l)))
        newinds = [Permutation.__call__(self, inds)]
        return [l[newinds[i]] for i in
