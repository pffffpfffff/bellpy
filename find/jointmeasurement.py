import numpy as np
import itertools as it
from bellpy.find.setting import *
from enum import Enum, unique

@unique
class Itmode(Enum):
    full = 1
    expect = 2
    ns = 3


class Joint_measurement:
    """
    Collect information about joint measurements
    """
    def __init__(self, settings, mode=Itmode.full):
        self.settings = sorted(settings, key= lambda x: x.party, reverse=False)
        self.parties = set([s.party for s in self.settings])
        # ensure that no two measurements occur on the same party
        if not( len(self.parties) == len(self.settings)):
            raise Exception('More than one measurement per party detected')

        """
        mode decides, how the iteration should happen
        Itmode.full: iterate over all combinations of outcomes
        Itmode.expect: iterate only over combinations that yield different
                expectation values
        """
        self.mode = mode
        self.iter = None
        if self.mode == Itmode.expect:
            self.iter = self.get_iter()
       #if self.mode==Itmode.full:
       #    raise Exception(self.mode)

    def devide_by_scenarios(self, scenario1, scenario2):
        assert scenario1.is_disjoint(scenario2)
        sett1 = []
        sett2 = []
        for s in self.settings:
            if s in scenario1:
                sett1.append(s)
            elif s in scenario2:
                sett2.append(s)
            else:
                raise Exception("Correlation could not be devided between\
                scenarios, setting not found")
        return Joint_measurement(sett1), Joint_measurement(sett2)

    def get_iter(self):
        itr = []
        evals = []
        jmit = JM_iterator(self)
        for jmversion in jmit:
            expval = jmversion.expectation_value() 
            if not ( expval in evals ):
                evals.append(expval)
                itr.append(jmversion)
        return itr


    def __iter__(self):
        if self.mode==Itmode.full:
            return JM_iterator(self)
        elif self.mode==Itmode.expect:
            return JM_iterator2(self)
        elif self.mode==Itmode.ns:
            return JM_nsiterator(self)
        raise Exception("Invalid mode for Joint Measurement")

    def __mul__(self, other):
        otherparties = set([s.party for s in other.settings])
        sett = self.settings + other.settings
        return Joint_measurement(sett, mode=self.mode)
    # # to avoid exceptions
      # if self.parties.isdisjoint(otherparties):
      #     sett = self.settings + other.settings
      #     return Joint_measurement(*sett)
      # else:
      #     return None
 
    @property
    def symbol(self):
        return "".join([s.symbol for s in self.settings])

    @property
    def outputs(self):
        return [s.value for s in self.settings]

    @property
    def inputs(self):
        return tuple([s.label for s in self.settings])

    def __str__(self):
        return self.symbol + ": {}".format(self.outputs)

    def expectation_value(self):
        return np.prod([s.value for s in self.settings])

       
        
class JM_iterator:
    def __init__(self, jm):
        self.jm = jm
        self.iter = it.product(*jm.settings)
    def __next__(self):
        return Joint_measurement(next(self.iter), mode=Itmode.full )

    def __iter__(self):
        return self

class JM_iterator2:
    def __init__(self, jm):
        self.jm = jm
        self.iter = iter(self.jm.iter)

    def __next__(self):
        return next(self.iter)

class JM_nsiterator:
    def __init__(self, jm):
        self.jm = jm
        self.settings = copy.deepcopy(self.jm.settings)
        self.remove_one_outcome()
        self.i = 0
        self.iter = it.product(*self.settings)

    def remove_one_outcome(self):
        for s in self.settings:
            if len(s.outcomes)>1:
                s.outcomes.pop(-1)
        return 0

    def __next__(self):
        ne = Joint_measurement(next(self.iter), mode=Itmode.full)
#       if self.i == 0:
#           try:
#               ne = Joint_measurement(next(self.iter), mode=Itmode.full)
#           except:
#               pass
#       self.i += 1
        return ne


