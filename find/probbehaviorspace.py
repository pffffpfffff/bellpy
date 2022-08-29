from bellpy.find.behavior import *
from bellpy.find.assignments import *
from bellpy.find.jointmeasurement import Itmode
from bellpy.find.function import Dual_vector
import bellpy.find.relabeling2 as rel
from bellpy.find.relabelings import Group

from sympy.combinatorics import SymmetricGroup, PermutationGroup, Permutation
import pdb
import numpy.linalg as la

class Probability_behavior_space(Behavior_space):

    def __init__(self, scenario:Scenario):
        self.sc = scenario
        self.check_scenario_sanity()
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

    def check_scenario_sanity(self):
        for s in self.sc.settings:
            if tuple(s.outcomes) != tuple(range(len(s.outcomes))):
                raise Exception("Invalid scenario supplied to \
Probability_behavior_space: All settings are required to have outcomes labeled \
consecutively starting with 0")
        return

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

    def relabelings_group(self, parties = None, settings = None, outcomes = None):
        groups = []
        if parties:
            groups.append(rel.Parties_group(SymmetricGroup(self.scenario.nparties),self))

        if settings:
            sbp = self.scenario.settings_by_party()
            triv_group = PermutationGroup(Permutation(0))
            for p in range(self.scenario.nparties):
                S = SymmetricGroup(len(sbp[p]) - 1)
                groups.append(rel.Settings_group(triv_group * S, self, p))
        if outcomes:
            for s in self.scenario.settings:
                groups.append(rel.Outcomes_group(self, s))

        if groups == []:
            return Group()
            
        return rel.Product_group(*groups)
            
            


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
        self.reconstruction = None
        self.compute_reconstruction()

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

    def array_for_label(self, label):
        v = np.zeros(self.dimension)
        v[self.labels.index(label)] = 1
        return v

    def compute_reconstruction(self):
        """
        returns dictionary: 
        """
        nparties = self.scenario.nparties
        rec = dict()

        for p in range(nparties + 1):
            fixlabel = str(self.fullspace.events[0])
            B = dict(zip(self.fullspace.labels, self.fullspace.labels))
            for e in self.fullspace.events:
                count, sparty = parties_with_last_outcome(e)
                label = str(e)
                if count == p:
                    if count == 0:
                        rec[label] = Dual_vector(self.array_for_label(label), string = "p[" + label + "]")
                    else:
                        marg, events = events_for_ns(self, e, sparty)
                        margfun = rec[str(marg)]
                        # NS constraints
                        rec[label] = margfun - sum([rec[str(x)] for x in events])

        self.reconstruction = np.array([rec[label].fs for label in self.fullspace.labels])
        return None

    def reconstruct_full(self, beh: Behavior_space_vector):
        return type(beh)(self.fullspace, np.dot(self.reconstruction, beh))

    def embed_in_fullspace(self, v: Behavior_space_vector):
        vfull = np.zeros(self.fullspace.dimension)
        vfull = type(v)(self.fullspace, vfull)
        for l in self.labels:
            vfull[l] = v[l]
        return vfull

    def behavior_vector_from_full(self, b: Behavior_space_vector):
       #self.indices = self.compute_indices()
        bns = [ b[l] for  l in self.labels]
        return type(b)(self, bns)

    @property
    def events(self):
        return [self.fullspace.events[i] for i in self.indices]

    def relabelings_group(self, parties=None, settings = None, outcomes = None):
        groups = []
        if parties:
            groups.append(rel.Parties_group(SymmetricGroup(self.scenario.nparties),self))

        if settings:
            sbp = self.scenario.settings_by_party()
            triv_group = PermutationGroup(Permutation(0))
            for p in range(self.scenario.nparties):
                S = SymmetricGroup(len(sbp[p]) - 1)
                groups.append(rel.Settings_group(triv_group * S, self, p))
        if outcomes:
            for s in self.scenario.settings:
                groups.append(rel.Outcomes_group(self, s))

        if groups == []:
            return Group()
            
        return rel.Product_group(*groups)
 
        

    def conversion_matrix_to_standard_basis(self, labels):
        indices = [self.fullspace.labels.index(e) for e in labels]
        rec_submatrix = [self.reconstruction[i] for i in indices]
        rec_submatrix = np.array(rec_submatrix, dtype=int)
        sh = np.shape(rec_submatrix)
        assert sh[0] == sh[1]
        assert la.matrix_rank(rec_submatrix) == sh[0]
        inv = la.inv(rec_submatrix)
        return inv

    def vector_in_standard_basis(self, b: Behavior_space_vector):
        inv = self.conversion_matrix_to_standard_basis(b.labels)
        if isinstance(b, "Bell_inequality"):
            return Bell_inequality(self, np.dot(inv.transpose(), b.table))
        if isinstance(b, "Behavior"):
            return Behavior(self, np.dot(inv, b.table))

