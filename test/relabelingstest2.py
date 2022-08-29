from bellpy import *
import numpy as np

A0 = Setting(0, 0, [0])
A1 = Setting(0, 1, [0,  1])
A2 = Setting(0, 2, [0,  1])
B0 = Setting(1, 0, [0])
B1 = Setting(1, 1, [0,  1])
B2 = Setting(1, 2, [0,  1])

scenario = Scenario([A0, A1, A2, B0, B1, B2])
jointmeas = scenario.joint_measurements()

bs = NSpb_space(scenario)
print('labels of the dimensions of the behavior space', bs.labels)
print('number of dimensions', bs.dimension)

ldm = Local_deterministic_model(bs)

#print(ldm.data_frame())
bis = ldm.bell_inequalities()
print('bis', bis)
chsh = bis[3]
#print('chsh', chsh )

print(bs.dimension)

def conversion_test(beh):
    print('labels')
    print(type(beh.behavior_space))
    print(beh.behavior_space.labels)
    print('1', beh, "\n")
    fullbeh = bs.reconstruct_full_behavior(beh)
    beh2 = bs.behavior_from_full_behavior(fullbeh)
    print('2', beh2)
    fullbeh2 = bs.reconstruct_full_behavior(beh2)
    print('-------------')
    print('full 1', fullbeh, "\n")
    print('full 2', fullbeh2)

    return 0

#jbeh = Behavior(bs, np.random.random(9))
G = bs.relabelings_group(setting = True, outcome = True)
g = next(iter(G))
#   conversion_test(beh)
#   conversion_test(g(beh))

#print('bis', bis)

chsh = bis[0]
print('chsh', chsh )
#   beh = Behavior(bs, [1] + [0]*0 +  [1] + [0]*7)
#   fb = bs.reconstruct_full_behavior(beh)
#   print(bs.reconstruction.keys())
#   for k, f in bs.reconstruction.items():
#       print(k, '\t', f.string) 
#   print('beh', beh)
#   print('full beh', fb)
for g in G:
    print(g(chsh))


