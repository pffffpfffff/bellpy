from bellpy import *
import time

A0 = Setting(0, 0, [0])
A1 = Setting(0, 1, [0,  1])
A2 = Setting(0, 2, [0,  1])
A3 = Setting(0, 3, [0,  1])
B0 = Setting(1, 0, [0])
B1 = Setting(1, 1, [0,  1])
B2 = Setting(1, 2, [0,  1])
B3 = Setting(1, 3, [0,  1])

scenario = Scenario([A0, A1, A2, A3, B0, B1, B2, B3])
bs = NSpb_space(scenario)
#fs = bs.fullspace
#   m = map(lambda x: str(x), fs.events)
#   print("fsevents", list(m) )


outG = Relabelings_group(SymmetricGroup(2), bs, "outcomes", setting = A1)
G = bs.relabelings_group(parties = True, settings = True, outcomes = True)
#   t1 = time.time()
#   for count, g in enumerate(G):
#       print(count)
#       print(np.shape(g.matrix))
#   t2 = time.time()
#   print(t2 - t1)

lhv = Local_deterministic_model(bs)
#   for b in lhv:
#       print(b)
bis = lhv.bell_inequalities()
bis.remove_duplicates(G)
print(bis)
#   print(bis.behavior_space)
#   print(bis.behavior_space.dimension)
#   print(next(iter(G)).matrix)
#   for g in outG:
#       print(g(fullbi))
#       print(".............")

