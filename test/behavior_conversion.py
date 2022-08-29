from bellpy import *
import numpy as np

A0 = Setting(0, 0, [0])
A1 = Setting(0, 1, [0,  1])
A2 = Setting(0, 2, [0,  1])
B0 = Setting(1, 0, [0])
B1 = Setting(1, 1, [0,  1])
B2 = Setting(1, 2, [0,  1])

scenario = Scenario([A0, A1, A2, B0, B1, B2])

bs = NSpb_space(scenario)
print('labels of the dimensions of the behavior space', bs.labels)
print('number of dimensions', bs.dimension)

vec = np.random.random(9)
print(vec)

beh = Behavior(bs, vec)
print(beh)
l =bs.labels[0]
print('l', l)
#.bs.reconstruction[l] = Function(lambda x: x[l])
print('reconstruction', bs.reconstruction[l])
B = dict(zip(bs.fullspace.labels, bs.fullspace.labels))
#   print('B', B)
#   #print(hash('A0B0: [0, 0]'), hash('A1B1: [1, 1]'))
#newb = bs.reconstruct_full_behavior(B)
#print('newb')
#print(newb)
fullbeh = bs.reconstruct_full_behavior(beh)
print(fullbeh)
for k, x in bs.reconstruction.items():
    print(k, '\t', x.string)


