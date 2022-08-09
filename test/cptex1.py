""" Find I3322 inequalities """
from bellpy import *

bs22 = Default_expectation_behavior_space(2,2)
print(bs22.labels)
# ['A0B0', 'A0B1', 'A0B2', 'A1B0', 'A1B1',
#  'A1B2', 'A2B0', 'A2B1', 'A2B2']

bs33 = Default_expectation_behavior_space(3,3)

chsh_array = [2, 0, 0, 0, -1, -1, 0, -1, 1]
chsh = Bell_inequality(bs22, chsh_array)

lhvm2222 = Local_deterministic_model(bs22)
lhvm3322 = Local_deterministic_model(bs33)

ext_map = Canonic_extension_map(bs22, (0,0,1), (1,0,1))
eb = Extended_behaviors(lhvm2222, chsh, ext_map)

cpt = Cone_projection_technique(lhvm3322, true_facets=True)
cpt.add_extended_behaviors(eb)
bi = cpt.bell_inequalities()
bi.remove_duplicates(party=True, setting=True, outcome=True)
print(bi)
