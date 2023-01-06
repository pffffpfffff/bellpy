import itertools as it
import abc
from bellpy.find.scenario import *
from bellpy.find.jointmeasurement import Itmode
from typing import List

class Assignments:
    def __init__(self, jmeasurements, scenario = None):
        self.jm = jmeasurements
        if isinstance(scenario, Scenario):
            self.scenario = scenario
        else:
            self.scenario = self.compute_scenario()

    @staticmethod
    def from_local_assignments(settings: List[Setting]):
        scenario = Scenario(settings) 
        jm = scenario.joint_measurements()
        return Assignments(jm, scenario = scenario)

    def compute_scenario(self):
        settings = []
        for jmeas in self.jm:
            for s in jmeas.settings:
                if not(s in settings):
                    settings.append(s)
        return Scenario(settings)


    def combine(self, other):
        return Assignments(self.jm + other.jm)

    def __mul__(self, other):
        assert self.scenario.is_disjoint(other.scenario)
        l = []
        for a0, a1 in it.product(self.jm, other.jm):
            if a0.parties.isdisjoint(a1.parties):
                l.append(a0*a1)
        return Assignments(l)

    @staticmethod
    def prod(list_of_assignments):
        a = list_of_assignments[0]
        for ass in list_of_assignments[1:]:
            a *= ass
        return a

    def __str__(self):
        return ", ".join([jmeas.__str__() for jmeas in self.jm])

    def __iter__(self):
        return iter(self.jm)
        


class Assignment_model(abc.ABC):
    """
    input: scenario
    output: one list of joint measurements with different assigned outputs at a
    time
    """

    @property
    @abc.abstractmethod
    def scenario(self):
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass

    def __len__(self):
        l = [x for x in self]
        return len(l)

class Unrestricted_assignments(Assignment_model):
    def __init__(self, scenario, mode=Itmode.full):
        self.sc = scenario
        self.mode = mode

    @property
    def scenario(self):
        return self.sc

    def __iter__(self):
        return Ua_iterator(self)

class Ua_iterator:
    def __init__(self, amodel: Assignment_model):
        self.model = amodel
        self.iter = it.product(*self.model.sc.joint_measurements(mode=self.model.mode))

    def __next__(self):
        return Assignments(next(self.iter))

class Local_deterministic_assignments(Assignment_model):
    def __init__(self, scenario:Scenario, partition=None, mode = Itmode.full):
        self.sc = scenario
        self.mode = mode
        if partition is None:
            partition = [[i] for i in range(self.scenario.nparties)]
        self.partition = partition

    @property
    def scenario(self):
        return self.sc

    def __iter__(self):
        return Lr_iterator(self)


class Lr_iterator:
    def __init__(self, lmodel: Local_deterministic_assignments):
        self.model = lmodel
        self.scenarios = [self.model.scenario.subscenario(p) for p in \
                self.model.partition]
        self.iter = it.product(*[Unrestricted_assignments(s,
            mode=self.model.mode) for s in \
            self.scenarios])

    def __next__(self):
        loa = next(self.iter)
        return Assignments.prod(loa)


