from bellpy import *

A0 = Setting(0, 0, [0])
A1 = Setting(0, 1, [0, 1])
A2 = Setting(0, 2, [0, 1])
A3 = Setting(0, 3, [0, 1])
B0 = Setting(1, 0, [0])
B1 = Setting(1, 1, [0, 1])
B2 = Setting(1, 2, [0, 1])
B3 = Setting(1, 3, [0, 1])


A0t = Setting(0, 0, [0])
A1t = Setting(0, 1, [0, 1, 2])
A2t = Setting(0, 2, [0, 1, 2])
A3t = Setting(0, 3, [0, 1, 2])
B0t = Setting(1, 0, [0])
B1t = Setting(1, 1, [0, 1, 2])
B2t = Setting(1, 2, [0, 1, 2])
B3t = Setting(1, 3, [0, 1, 2])

f = lambda x: int(x>=1)
c1 = Coarse_graining(A0t, A0, lambda x: x)
c2 = Coarse_graining(A1t, A1, f)
c3 = Coarse_graining(A2t, A2, f)
c4 = Coarse_graining(B0t, B0, lambda x: x)
c5 = Coarse_graining(B1t, B1, f)
c6 = Coarse_graining(B2t, B2, f)
c7 = Coarse_graining(A3t, A3, f)
c8 = Coarse_graining(B3t, B3, f)

# third argument maps outcomes of B2t to B2, should be surjective

original_scenario = Scenario([A0, A1, A2, A3, B0, B1, B2, B3])
target_scenario = Scenario([A0t, A1t, A2t, A3t, B0t, B1t, B2t, B3t])

original_bs = NSpb_space(original_scenario)
print("lab ori", original_bs.labels)
target_bs = NSpb_space(target_scenario)
print("lab tar", target_bs.labels)
print(len(target_bs.labels))

infl = Inflation(original_bs, target_bs, c1, c2, c3, c4, c5, c6, c7, c8)

ldm = Local_deterministic_model(original_bs)
bis = ldm.bell_inequalities()
print('bi', bis[0])
i3322 = bis[0]
#   chsh = bis[3]
#   print('chsh', chsh)

extbeh = Extended_behaviors(ldm, i3322, infl)

ldm_target = Local_deterministic_model(target_bs)

#   print(ldm_target)
#   bis = ldm_target.bell_inequalities()
#   bis.remove_duplicates(party = True, setting = True, outcome = True)

#   print('bis', bis)

cpt = Cone_projection_technique(ldm_target)

cpt.add_extended_behaviors(extbeh)
cpt.add_party_symmetry()
bis = cpt.bell_inequalities(interactive=True)

#bis.remove_duplicates(party = True, setting = True, outcome = True) 
print("////////////////////////////////")
for bi in bis:
    print("....")
    print(bi)

#   allbis = ldm_target.bell_inequalities()
#   allbis.remove_duplicates(party = True, setting = True, outcome = True)
#   print("all bis ////////////////////////////////")
#   for bi in allbis:
#       print("....")
#       print(bi)



