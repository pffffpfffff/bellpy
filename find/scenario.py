import itertools as it
from bellpy.find.jointmeasurement import *
from bellpy.find.setting import *
from typing import List

class Scenario:
    def __init__(self, settings: List[Setting]):
        self.settings = settings

    @property
    def parties(self):
        return list(set([s.party for s in self.settings]))

    @property
    def nparties(self):
        return max(self.parties)+1

    @property
    def nsettings(self):
        nsett = [0]*self.nparties
        for s in self.settings:
            nsett[s.party] += 1
        return nsett

    def is_disjoint(self, other):
        return all([s[0] != s[1] for s in it.product(self.settings,\
            other.settings)])

    def __add__(self, other):
        assert self.is_disjoint(other)
        return Scenario(self.settings + other.settings)

    def settings_by_party(self):
        p = self.parties
        sbp = []
        for par in p:
            sp = []
            for s in self.settings:
                if s.party == par:
                    sp.append(s)
            sbp.append(sp)
        return sbp

    def input_of_party(self, inp, party):
        for s in self.settings:
            if s.party == party and s.label == inp:
                return s
        raise Exception("Input " + str(inp) + " does not exist for party " +
                str(party))

    def joint_measurements(self, mode=Itmode.full):
        sbp = self.settings_by_party()
        jm = [Joint_measurement(c, mode=mode) for c in it.product(*sbp)]
        return jm

    def subscenario(self, parties):
        sett = []
        for s in self.settings:
            if s.party in parties:
                sett.append(s)
        return Scenario(sett)


