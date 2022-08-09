from bellpy.find.assignments import *
from bellpy.find.scenario import *
import bellpy.find.partysym as psym
import bellpy.find.relabelings as rel
from typing import List
from bellpy.find.jointmeasurement import Itmode

class Behavior_space(abc.ABC):

    @property
    @abc.abstractmethod
    def scenario(self):
        pass

#   @property
#   @abc.abstractmethod
#   def dimension(self):
#       pass

    @property
    @abc.abstractmethod
    def mode(self):
        # either Itmode.full, Itmode.expect, or Itmode.ns (nosignaling)
        pass

    @property
    @abc.abstractmethod
    def labels(self): 
        pass

    @property
    def dimension(self):
        return len(self.labels)

    def array_to_behavior(self, arr):
        return Behavior(self, arr)

    @abc.abstractmethod
    def behavior_from_assignments(self, assignments: Assignments):
        pass

#   @abc.abstractmethod
#   def array_to_canonic_array(self, arr):
#       pass

#   @abc.abstractmethod
#   def canonic_array_to_array(self, arr):
#       pass

    def facet_to_bell_inequality(self, facet):
        return Bell_inequality(self, facet)


    def facets_to_bell_inequalities(self, facets):
        inequalities = []
        for f in facets:
            bi = self.facet_to_bell_inequality(f)
            inequalities.append(bi)
        return Bell_inequality_collection(inequalities)



class Expectation_behavior_space(Behavior_space):
    """
    Elements are expectation values of joint measurements given by scenario
    """
    def __init__(self, nsettings):
        self.nsettings = nsettings
        # trivial setting 0 on each party not counted
        self.sc = self.compute_scenario()
        self.labls = self.compute_labels()


    @property
    def mode(self):
        # set how to iterate over assignments in this space: enough to consider
        # different expectation values?
        return Itmode.expect


    @abc.abstractmethod
    def compute_scenario(self) -> Scenario:
        pass


    @property
    @abc.abstractmethod
    def tuples(self):
        pass

    def array_to_canonic_array(self, arr):
        shape = self.scenario.nsettings
        canarr = np.zeros(shape)
        correlation_values = iter(arr)
        for t in self.tuples:
            canarr[t] = next(correlation_values)
        return canarr

    def canonic_array_to_array(self, carr):
        shape = self.scenario.nsettings
        arr = [carr[t] for t in self.tuples]
        return arr

    def party_symmetry(self):
        return psym.party_symmetry(self.tuples)

    def no_margs(self):
        return psym.no_margs(self.tuples)

    def only_nbodymargs(self, n, mode='le'):
        return psym.only_nbodymargs(self.tuples, n, mode)

    def full_body_correlations(self):
        return self.only_nbodymargs(len(self.nsettings), mode='eq')

    def correlation_value_by_tuple(self, behavior, tup):
        t = self.tuples
        dct = { t[i]: behavior[i] for i in range(self.dim) }
        return dct[tup]

    def party_permutation_group(self):
        return rel.PartyGroup(self.tuples)

    def settings_permutation_group(self, parties=None):
        if parties is None:
            parties = range(self.scenario.nparties)
        s = [rel.SettingGroup(self.tuples, i) for i in parties]
        return rel.mdot(*s)

    def outcomes_permutation_group(self):
        return rel.OutcomeGroup(self.tuples)


class Default_expectation_behavior_space(Expectation_behavior_space):
    """
    Elements are expectation values of joint measurements given by scenario
    Marginals are included
    """
    def __init__(self, *nsettings: List[int]):
        self.nsettings = nsettings
        # trivial setting 0 on each party not counted
        self.sc = self.compute_scenario()
        self.labls = self.compute_labels()

    @property
    def labels(self):
        return self.labls

    @property
    def scenario(self):
        return self.sc

    @property
    def dimension(self):
        return len(self.labels)

    def compute_scenario(self) -> Scenario:
        settings = []
        party = 0
        for n in self.nsettings:
            settings += [Setting(party, 0, [1])] + \
                    [Setting(party, i+1, [-1,1]) for i in range(n)]
            party += 1
        return Scenario(settings)

    def compute_labels(self):
        jmeass = self.scenario.joint_measurements()
        labs = [ jm.symbol for jm in jmeass ]
        return labs

    def behavior_from_assignments(self, assignments):
        assdct = { a.symbol: np.prod(a.outputs) for a in assignments}
       #print(assdct)
        tab = [ assdct[l] for l in self.labels]
        return Behavior(self, tab)

    @property
    def tuples(self):
        jmeass = self.scenario.joint_measurements()
        tups = []
        for jm in jmeass:
            t = ["None"]*self.scenario.nparties
            for sett in jm.settings:
                t[sett.party] = sett.label
            tups.append(tuple(t))
        return tups

    def array_to_canonic_array(self, arr):
        shape = [s + 1 for s in self.nsettings]
        return np.reshape(arr, shape)

class Custom_expectation_behavior_space(Expectation_behavior_space):
    """
    Elements are expectation values of joint measurements given by scenario
    Which correlations are considered is determined by tuples (0,2,1) =
    expect(A_0 B_2 C_1)

    By default, the 0th setting has only one outcome labeled 1

    corrlist should not include the correlation (0,0,0), this one is included
    automatically as first correlation
    """
    def __init__(self, corrlist):
        self.corrlist = corrlist
        self.sc = self.compute_scenario()
        self.labls = self.compute_labels()


    @property
    def mode(self):
        return Itmode.expect

    @property
    def labels(self):
        return self.labls

    @property
    def scenario(self):
        return self.sc

    @property
    def dimension(self):
        return len(self.labels)

    def array_to_canonic_array(self, arr):
        shape = [s + 1 for s in self.scenario.nsettings]
        canarr = np.zeros(shape)
        correlation_values = iter(arr)
        for t in self.tuples:
            canarr[t] = next(correlation_values)
        return canarr

    def compute_scenario(self):
        settings = []
        setting_tracker = { pty: [] for pty in range(len(self.corrlist[0])) }
        for corr in self.corrlist:
            party = 0
            for m in corr:
                if not (m in setting_tracker[party]):
                    if m == 0:
                        settings.append(Setting(party, 0, [1]))
                        setting_tracker[party].append(m)
                    else:
                        settings.append(Setting(party, m, [-1,1]))
                        setting_tracker[party].append(m)
                party += 1
        return Scenario(settings)

    def compute_labels(self):
        return [ str(c) for c in self.tuples ] 

    def behavior_from_assignments(self, assignments):
        assdct = { a.inputs: np.prod(a.outputs) for a in assignments}
        tab = [1] + [ assdct[l] for l in self.corrlist]
        return Behavior(self, tab)

    @property
    def tuples(self):
        nparties = len(self.corrlist[0])
        return [tuple([0]*nparties)] + self.corrlist





class Behavior:
    def __init__(self, bs: Behavior_space, table):
        self.table = table
        self.bs = bs
        self.label_dct = dict(zip(self.bs.labels, self.table))

    @property
    def behavior_space(self):
        return self.bs

    def __str__(self):
        a = self.table
        signs = [np.sign(x) for x in a]
        signs = [ " +" if s==1 else " -" for s in signs]
        a = np.absolute(a)
        l = list(zip(signs, a, self.bs.labels))
        st = "".join([ x[0] + "{} ".format(x[1]) + x[2] for x in l])
        return st

    def to_canonic_array(self):
        # output array that is compatible with inequality library
        return self.bs.array_to_canonic_array(self.table)

    def __getitem__(self, i):
        if type(i) is int:
            return self.table[i]
        else:
            try:
                return self.label_dct[i]
            except:
                raise Exception("Cannot retrieve coefficient for key or index "
                        + str(i))


#   def to_array(self):
#       return self.table

#   class Expval_behavior(Behavior):
#       def __call__(self, tup):
#           return self.bs.correlation_value_by_tuple(self, tup)


class Bell_inequality:
    def __init__(self, bs: Behavior_space, table):
        self.table = table
        self.bs = bs

    @property
    def behavior_space(self):
        return self.bs

    def __str__(self):
        a = self.table
        signs = [np.sign(x) for x in a]
        signs = [ " +" if s==1 else " -" for s in signs]
        a = np.absolute(a)
        l = list(zip(signs, a, self.bs.labels))
        st = "".join([ x[0] + "{} ".format(x[1]) + x[2] for x in l if x[1]!=0])
        return st

    def to_canonic_array(self):
        # output array that is compatible with inequality library
        return self.bs.array_to_canonic_array(self.table)

    def __getitem__(self, i):
        return self.table[i]

    def __call__(self, beh: Behavior):
        if not ( beh.behavior_space == self.behavior_space ):
            raise ValueError('Behavior is incompatible with Bell inequality')
        return np.dot(self.table, beh.table)

class Bell_inequality_collection:
    def __init__(self, bell_inequalities):
        self.bis = bell_inequalities
        self.behavior_space = self.check_sanity()

    def __iter__(self):
        return iter(self.bis)

    def check_sanity(self):
        bs = self.bis[0].behavior_space
        for bi in self.bis:
            if not ( bi.behavior_space is bs ):
                raise ValueError('Bell inequalities in \
                Bell_inequality_collection must have the same Behavior_space')
        return bs

    def remove_duplicates(self, party = None, setting = None, outcome = None):
        
        l = [tuple(bi.table) for bi in self]
        uniq_l = list(set(l))
        inds1 = [ l.index(x) for x in uniq_l ]

        if (not party) and (not setting ) and (not outcome):
            l = [list(t) for t in uniq_l]
            l = [ Bell_inequality(self.behavior_space, t) for t in l ]
            self.bis = l
            return inds1

        groups = []
        if party:
            groups.append(self.behavior_space.party_permutation_group())
        if setting:
            groups += [rel.SettingGroup(self.behavior_space.tuples, i) for i in
                    range(self.behavior_space.scenario.nparties)]
        if outcome:
            groups.append(self.behavior_space.outcomes_permutation_group())
        G = rel.mdot(*groups)

        l = [np.array(t) for t in l]

        new_bi_tabs, inds = G.equivalence_classes(l, inds1) 

        self.bis = [ Bell_inequality(self.behavior_space, t) for t in
                new_bi_tabs ]

        return inds

    def __str__(self):
        return '\n'.join([str(bi) for bi in self.bis])

    def __len__(self):
        return len(self.bis)

    def __getitem__(self, i):
        return self.bis[i]


