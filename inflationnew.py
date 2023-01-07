
from bellpy.find.scenario import *
from bellpy.find.jointmeasurement import *
from bellpy.find.cpt import Extension_map
from bellpy.find.behavior import *

import copy
from typing import Tuple, List

class Inflation_edge:
    def __init__(self, original_setting: Setting, target_setting: Setting):
        self.original: Setting = original_setting 
        self.target: Setting = target_setting 

        self.fine_grain = { o: None for o in self.original.outcomes }

    def set_original_outcome(self, orig_out):
        assert orig_out in self.original.outcomes
        self.original_outcome = orig_out
        
    def set_target_outcome(self, or_out, targ_out):
        assert targ_out in self.target.outcomes
        assert or_out in self.original.outcomes
        self.fine_grain[or_out] = tar_out

    def check_injective(self):
        if not len(self.fine_grain.values()) == len(set(self.fine_grain.values())):
            raise Exception(f"Fine graining {self} is not injective")

    def __str__(self):
        return f"{self.original} -> {self.target}"

    def check_defines_fine_graining(self):
        for o, tar_o in self.fine_grain.items():
            if tar_o is None:
                raise Exception(f"Set a target outcome for outcome {o} of setting {self.original}.")
            if not tar_o in self.target.outcomes:
                raise Exception(f"{tar_o} is not a valid outcome of measurement {self.target}.")

    def __call__(self, event: Event):
        """ returns new Event where (original_outcome | original) is replaced by
            (target_outcome | target). """
        if self.original_outcome is None or self.target_outcome is None:
            raise Exception("set_original_outcome and set_target_outcome before using this method.")
        # Implement this function after refactoring of events.
        pass

class Inflation:
    def __init__(self, original: Behavior_space, target: Behavior_space, *ie: Tuple[Inflation_edge, ...]):
        self.obs = original # original behavior space
        self.tbs = target # target behavior_space
        self.edges = { s: None for s in self.tbs.scenario }
        self.check_inflation_edges(ie)
        for inflation_edge in ie:
            self.edges[inflation_edge.target] = inflation_edge

    def check_inflation_edges(self, iedges: Tuple[Inflation_edge, ...]):
        """
        Two conditions must be satisfied:
        1) All settings must be in the scenarios of the inflation
            1. a) For any inflation edge ie, ie.original in self.original.scenario 
            1. b) For any inflation edge ie, ie.target in self.target.scenario 

        2) All settings in self.target.scenario have degree 1.
            2. a) degree lower equal 1
            2. b) degree greater equal 1
        """

        # 1. a)
        for ie in iedges:
            if not ie.original in self.original.scenario:
                raise Exception(f"{ie.original} is not in original scenario") 

        # 1. b)
        for ie in iedges:
            if not ie.target in self.target.scenario:
                raise Exception(f"{ie.target} is not in target scenario") 

        # 2. a)
        target_settings = [ie.target for ie in iedges]
        deg_lower_equal_1 = ( len(target_settings) == len(set(target_settings)))
        if not deg_lower_equal_1:
            raise Exception("Invalid definition of Inflation: Settings from
target scenario can appear in at most one edge.")

        # 2. b)
        target_settings = set(target_settings)
        for s in self.target.scenario:
            if not s in self.target_settings:
                raise Exception(f"No edge found for target setting {s}")            

        return None

   
