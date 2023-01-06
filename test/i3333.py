from bellpy import *

"""
Find generalizations of i3322 to three settings
"""

A0 = Setting(0, 0, [0])
A1 = Setting(0, 1, [0, 1])
A2 = Setting(0, 2, [0, 1])
A3 = Setting(0, 3, [0, 1])

B0 = Setting(1, 0, [0])
B1 = Setting(1, 1, [0, 1])
B2 = Setting(1, 2, [0, 1])
B3 = Setting(1, 3, [0, 1])

i3322_scenario = Scenario([A0, A1, A2, A3, B0, B1, B2, B3])
i3322_space = NSpb_space(i3322_scenario)
i3322_lhvm = Local_deterministic_model(i3322_space)

#   i3322 = i3322_lhvm.bell_inequalities()[3]
#   print(i3322.table)
i3322 = Bell_inequality(i3322_space, [1, 0, -1, 1, -1, 1, 1, -1, 1, -1, 0, -1, 0, -1, 1, 1])

A0t = Setting(0, 0, [0])
A1t = Setting(0, 1, [0, 1, 2])
A2t = Setting(0, 2, [0, 1, 2])
A3t = Setting(0, 3, [0, 1, 2])

B0t = Setting(1, 0, [0])
B1t = Setting(1, 1, [0, 1, 2])
B2t = Setting(1, 2, [0, 1, 2])
B3t = Setting(1, 3, [0, 1, 2])

i3333_scenario = Scenario([A0t, A1t, A2t, A3t, B0t, B1t, B2t, B3t])
i3333_space = NSpb_space(i3333_scenario)
i3333_lhvm = Local_deterministic_model(i3333_space)

idd = lambda x: x
fun = lambda x: int(x>1)

# now define coarse grainings

cg0 = Coarse_graining(A0t, A0, idd) 
cg1 = Coarse_graining(A1t, A1, fun) 
cg2 = Coarse_graining(A2t, A2, fun) 
cg3 = Coarse_graining(A3t, A3, fun) 

cg4 = Coarse_graining(B0t, B0, idd) 
cg5 = Coarse_graining(B1t, B1, fun) 
cg6 = Coarse_graining(B2t, B2, fun) 
cg7 = Coarse_graining(B3t, B3, fun) 


inflation = Inflation(i3322_space, i3333_space, cg0, cg1, cg2, cg3, cg4, cg5,
cg6, cg7)

extension_maps = inflation.extension_maps()

cpt = Cone_projection_technique(i3333_lhvm, true_facets = True)
for em in extension_maps[0:]:
    cpt.add_extended_behaviors(Extended_behaviors(i3322_lhvm, i3322, em)) 

#cpt.add_party_symmetry()

bell_inequalities = cpt.bell_inequalities(interactive = True)
print('i3322', i3322)
print('generalizations', bell_inequalities)
i3333 = bell_inequalities[0]

i3333def = inflation.deflate(i3333)
print(i3333def == i3322)
print(i3333def)

