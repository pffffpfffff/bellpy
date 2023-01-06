
from bellpy.find.scenario import *
from bellpy.find.jointmeasurement import *
from bellpy.find.cpt import Extension_map
from bellpy.find.behavior import *

import copy
from typing import Tuple, List

class Coarse_graining:
    def __init__(self, new: Setting, old: Setting, fun):
        self.new = new
        self.old = old
        self.fun = fun
        self.check_surjective()

    def check_surjective(self):
        """
        self.fun has to be a surjective map from new.outcomes --> old.outcomes
        """
        sur =( tuple(set([self.fun(o) for o in self.new.outcomes])) == tuple(self.old.outcomes))
        if not sur:
            raise Exception("Coarse graining function: new.outcomes --> old.outcomes \
must be surjective because new measurement is complete.")
        return None

    def __call__(self, event: Joint_measurement):
        settings = copy.copy(event.settings)
        for i, s in enumerate(settings):
            if s == self.new:
                value = self.fun(s.value)
                s = copy.copy(self.old)
                s.value = value
                settings[i] = s
        return Joint_measurement(settings) 

    def fine_grainings(self):
        cg_dict = {o: [] for o in self.old.outcomes}
        for to in self.new.outcomes:
            cg_dict[self.fun(to)].append(to)
        maps_to = it.product(*(cg_dict.values()))
        fine_gs = [Fine_graining(self.new, self.old, dict(zip(self.old.outcomes,
images))) for images in maps_to] 
        return fine_gs
        

class Fine_graining:
    def __init__(self, new: Setting, old: Setting, map_dict):
        self.new = new
        self.old = old
        self.map_dict = map_dict
        self.fun = lambda key: self.map_dict[key]

    def __call__(self, event: Joint_measurement):
        """
        in event, map the old measurement from the original scenario
        to a new measurement from the target scenario with a value, such that
        coarse graining would yield the original event again
        """
        settings = copy.copy(event.settings)
        for i, s in enumerate(settings):
            if s == self.old:
                value = self.fun(s.value)
                s = copy.copy(self.new)
                s.value = value
                settings[i] = s
        return Joint_measurement(settings) 

    def __str__(self):
        return f"{self.old.symbol} -> {self.new.symbol}, {self.map_dict}"


class Fine_grain_extension_map(Extension_map):
    def __init__(self, domain, codomain, fine_grainings: List[Fine_graining]):
        # for each edge (coarse graining) in the inflation, choose exactly one
        # fine graining
        # organize fine grainings in dict, where old settings are keys
        self.fgs = self.compute_finegraindict(fine_grainings)
        self._domain = domain # original behavior space
        self._codomain = codomain # target behavior space
        self.matrix = None
        self.compute_matrix()

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    def compute_finegraindict(self, fine_grainings: List[Fine_graining]):
        dct = {}
        for fg in fine_grainings:
            if fg.old in dct:
                dct[fg.old].append(fg)
            else:
                dct[fg.old] = [fg] 
        return dct

    def targets_for_event(self, event):
        targets = [event]
        for oldsetting, olds_finegrainings in self.fgs.items():
            newtargets = [ fg(e) for e in targets for fg in olds_finegrainings ]
            targets = newtargets
        return list(set(targets))

    def events_to_vector(self, events):
        beh = Behavior(self.codomain, np.zeros(self.codomain.dimension))
        for e in events:
            if str(e) in beh.labels:
                beh[str(e)] = 1
        return beh.table
        
    def compute_matrix(self):
        matrix = []
       #print('events', self.domain.events())
        for e in self.domain.fullspace.events:
            targets = self.targets_for_event(e)
            v = self.events_to_vector(targets)
            matrix.append(v)
        self.matrix = np.transpose(matrix)
        return 0
    
    def __call__(self, beh: Behavior):
        assert beh.behavior_space == self.domain
        fullbeh = beh.behavior_space.reconstruct_full(beh)
        return Behavior(self.codomain, np.dot(self.matrix, fullbeh.table))

    def __str__(self):
        return str([[str(x) for x in v] for k, v in self.fgs.items()])
        
        

class Inflation:
    def __init__(self, original: Behavior_space, target: Behavior_space, *cg: Tuple[Coarse_graining, ...]):
        self.obs = original # original behavior space
        self.tbs = target # target behavior_space
        self.cgs = cg
        self.check_degree()

    def check_degree(self):
        """
        every measurment in the new scenario should appear exactly once in a
        Coarse_graining
        """
        newsettings = [cg.new for cg in self.cgs]
        deg = ( len(newsettings) == len(set(newsettings)))
        if not deg:
            raise Exception("Invalid definition of Inflation: Every setting in target scenario\
should be the coarse graining of exactly one setting in the original scenario")
        return None

    def target_scenario(self):
        return Scenario([cg.new for cg in self.cgs])

    def original_scenario(self):
        return Scenario([cg.old for cg in self.cgs])

       #newb = Behavior(self.tbs, np.zeros(self.tbs.dimension))
       #for e in self.tbs.events:
       #    oev = self(e)
       #    newb[str(e)] = arg * Behavior(self.obs, self.obs.vector(oev))
       #return newb



    def coarse_grain(self, arg):
        if isinstance(arg, Joint_measurement):
            e = copy.copy(arg)
            for cg in self.cgs:
                e = cg(e)
            return e

        if isinstance(arg, Behavior):
            assert arg.behavior_space == self.tbs
           #beh_original = Behavior(self.obs, np.zeros(self.obs.dimension))
            full_target_behavior = self.tbs.reconstruct_full(arg)
            full_original_behavior = \
                Behavior(self.obs.fullspace, np.zeros(self.obs.fullspace.dimension))
            for e in full_target_behavior.events:
                coarse_grained_event = self.coarse_grain(e)
                full_original_behavior[str(coarse_grained_event)] += full_target_behavior[str(e)]
            return self.obs.behavior_vector_from_full(full_original_behavior)
                  
        
           #for oev in self.obs.events:
           #    if arg[oev] == 0:
           #    

    
    def extension_maps(self):
        all_fine_grainings = [ cg.fine_grainings() for cg in self.cgs ]
        return [ Fine_grain_extension_map(self.obs, self.tbs, fgs) for fgs in
it.product(*all_fine_grainings)] 
        
         
    def deflate(self, bi: Bell_inequality, em: Extension_map):
        raise Exception("This method gives a wrong result if several settings in
target behavior space generalize the same setting in the original one")
    
        assert bi.behavior_space is self.tbs
        bell_deflated = type(bi)(self.obs, np.zeros(self.obs.dimension))
        for e in bi.events:
            coarse_grained_event = self.coarse_grain(e)
            bell_deflated.table += self.obs.vector(coarse_grained_event) * bi[str(e)]
        return bell_deflated
    
