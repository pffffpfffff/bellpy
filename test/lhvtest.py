from bellpy import *
from sympy.combinatorics import SymmetricGroup

class Lhvm(Model):
    def __init__(self, beh: Behavior):
        self.bs = beh.behavior_space
        self.G = self.behavior_space.relabelings_group(parties = True,
settings = True, outcomes = True)
        self.Giter = None
        self.beh = beh

    @property
    def behavior_space(self):
        return self.bs
        
    def __iter__(self):
        self.Giter = iter(self.G)
        return self

    def __next__(self):
        return next(self.Giter)(self.beh)

A0 = Setting(0, 0, [0])
A1 = Setting(0, 1, [0,  1])
A2 = Setting(0, 2, [0,  1])
B0 = Setting(1, 0, [0])
B1 = Setting(1, 1, [0,  1])
B2 = Setting(1, 2, [0,  1])

scenario = Scenario([A0, A1, A2, B0, B1, B2])
bs = NSpb_space(scenario)

fs = bs.fullspace

b0 = Behavior(bs, [1] + [0]*8)
bfull = bs.embed_in_fullspace(b0)
bfull["A1B0: [1, 0]"] = 1
ldm = Lhvm(bfull)
ldm2 = Local_deterministic_model(fs)
bis = ldm.bell_inequalities()
bis2 = ldm2.bell_inequalities()
#   print(bis)
#   print('bis2', bis2)
print('ldm')
for b in ldm:
    print(b)
print('ldm 2', ldm2)


