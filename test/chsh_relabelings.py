from bellpy import *

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

bis.remove_duplicates(party=True, setting=True, outcome=True)
print("wo duplicates", bis)



