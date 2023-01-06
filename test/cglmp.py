from bellpy import *

"""
Find generalizations of CHSH to the cglmp scenario
"""

A0 = Setting(0, 0, [0])
A1 = Setting(0, 1, [0, 1])
A2 = Setting(0, 2, [0, 1])

B0 = Setting(1, 0, [0])
B1 = Setting(1, 1, [0, 1])
B2 = Setting(1, 2, [0, 1])

chsh_scenario = Scenario([A0, A1, A2, B0, B1, B2])
chsh_space = NSpb_space(chsh_scenario)
chsh_lhvm = Local_deterministic_model(chsh_space)
#   chsh = chsh_lhvm.bell_inequalities()[3]
#   print(chsh.table)
#   chsh = Bell_inequality(chsh_space,[1, 0, -1, 0, -1, 1, -1, 1, 1])
chsh = Bell_inequality(chsh_space,[ 0,  1,  0,  1, -1, -1,  0, -1,  1])

A0t = Setting(0, 0, [0])
A1t = Setting(0, 1, [0, 1, 2])
A2t = Setting(0, 2, [0, 1, 2])

B0t = Setting(1, 0, [0])
B1t = Setting(1, 1, [0, 1, 2])
B2t = Setting(1, 2, [0, 1, 2])

cglmp_scenario = Scenario([A0t, A1t, A2t, B0t, B1t, B2t])
cglmp_space = NSpb_space(cglmp_scenario)
cglmp_lhvm = Local_deterministic_model(cglmp_space)

idd = lambda x: x
fun = lambda x: int(x>=1)

# now define coarse grainings

cg0 = Coarse_graining(A0t, A0, idd) 
cg1 = Coarse_graining(A1t, A1, fun) 
cg2 = Coarse_graining(A2t, A2, fun) 
cg3 = Coarse_graining(B1t, B1, fun) 
cg4 = Coarse_graining(B2t, B2, fun) 
cg5 = Coarse_graining(B0t, B0, idd) 


inflation = Inflation(chsh_space, cglmp_space, cg0, cg1, cg2, cg3, cg4, cg5)

extension_maps = inflation.extension_maps()

cpt = Cone_projection_technique(cglmp_lhvm, true_facets = False)
for em in extension_maps[0:1]:
    cpt.add_extended_behaviors(Extended_behaviors(chsh_lhvm, chsh, em)) 

#cpt.add_party_symmetry()

bell_inequalities = cpt.bell_inequalities()
print('chsh', chsh)
print('generalizations', bell_inequalities)


