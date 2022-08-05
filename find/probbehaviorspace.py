from bellpy.find.behavior import *
from bellpy.find.assignments import *
from bellpy.find.jointmeasurement import Itmode
from bellpy.find.function import Function

class Probability_behavior_space(Behavior_space):

    def __init__(self, scenario:Scenario):
        self.sc = scenario
        self.jms = self.sc.joint_measurements()
        self.events = self.compute_events()


    @property
    def scenario(self):
        return self.sc

    @property
    def dimension(self):
        return len(self.labels)


    @property
    def mode(self):
        return Itmode.full

    @property
    def labels(self): 
        return [x.__str__() for x in self.events]

    def compute_events(self):
        l = []
        for jm in self.jms:
            for outjoint in jm:
                l.append(outjoint)
        return l

    def behavior_from_assignments(self, assignments: Assignments):
        b = []
        for jm in assignments:
            for outjoint in jm:
                if outjoint.outputs == jm.outputs:
                    b.append(1)
                else:
                    b.append(0)
        return Behavior(self, b)


def event_last_outcome(jm: Joint_measurement, party: int):
    s = jm.settings[party]
    assert s.party == party
    if len(s.outcomes)<2:
        return False
    if s.value == s.outcomes[-1]:
        return True
    return False

def parties_with_last_outcome(jm: Joint_measurement):
    count = 0
    smallest_party = 0
    for p in range(len(jm.settings)):
        if event_last_outcome(jm, p):
            count += 1
            if count == 1:
                smallest_party = p
    return count, smallest_party

def events_for_ns(bspace: Behavior_space, jm: Joint_measurement, party):
    s = jm.settings[party]
    assert s.party == party
    settings = copy.deepcopy(jm.settings)
    out = s.value

    trivsett = bspace.scenario.input_of_party(0, party)
    nsettings = copy.copy(settings)
    nsettings[party] = trivsett
    margevent = Joint_measurement(nsettings) 


    events = []
    for o in s.outcomes:
        if o!=s.value:
            newsett = copy.copy(s)
            newsett.value = o
            nsettings = copy.copy(settings)
            nsettings[party] = newsett
            events.append(Joint_measurement(nsettings))
    return margevent, events



class NSpb_space(Behavior_space):
    # No signaling behavior space with conditional probabilities as correlations
    def __init__(self, scenario: Scenario):
        self.fullspace = Probability_behavior_space(scenario)
        self.llables = self.compute_labels()
        self.indices = self.compute_indices()
        self.reconstruction = self.compute_reconstruction()

    @property
    def scenario(self):
        return self.fullspace.scenario

    @property
    def mode(self):
        return Itmode.full

    @property
    def labels(self):
        return self.llables

    def compute_labels(self):
        l = []
        for jm in self.fullspace.jms:
            jm.mode = Itmode.ns
            for outjoint in jm:
                l.append(outjoint.__str__())
        for jm in self.fullspace.jms:
            jm.mode = Itmode.full
        return l

    @property
    def dimension(self):
        return len(self.labels)

    def compute_indices(self):
        l1 = self.fullspace.labels
        l2 = self.labels
        inds = [l1.index(x) for x in l2]
        return inds

    def behavior_from_assignments(self, assignments: Assignments):
        b = self.fullspace.behavior_from_assignments(assignments).table
        bnew = [b[x] for x in self.indices]
        return Behavior(self, bnew)

    def compute_reconstruction(self):
        """
        returns dictionary: 
        key: label of full behavior space, 
        value: function: behavior -> correlation for label
        """
        nparties = self.scenario.nparties
        rec = dict()
        for p in range(nparties):
            for e in self.fullspace.events:
                count, sparty = parties_with_last_outcome(e)
                if count == p:
                    if count == 0:
                        rec[str(e)] = Function(lambda x: x[str(e)])
                    else:
                        marg, events = events_for_ns(self, e, sparty)
                        margfun = Function(lambda x: x[str(marg)])
                        # NS constraints
                        rec[str(e)] = margfun - sum([rec[str(x)] for x in events])
        return rec

    def reconstruct_full_behavior(self, beh: Behavior):
        cache = dict()
        fullbeh = []
        for l in self.fullspace.labels:
            if l not in cache:
                c = self.reconstruction[l](beh) 
                cache[l] = c
            fullbeh.append(cache[l])
        return Behavior(self.fullspace, fullbeh)

    def behavior_from_full_behavior(self, b: Behavior):
        bns = [ b[i] for  i in self.indices]
        return Behavior(self, bns)

