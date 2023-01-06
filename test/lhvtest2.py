from bellpy import *


A0 = Setting(0, 0, [0])
A1 = Setting(0, 1, [0, 1])
A2 = Setting(0, 2, [0, 1])

B0 = Setting(1, 0, [0])
B1 = Setting(1, 1, [0, 1])
B2 = Setting(1, 2, [0, 1])

chsh_scenario = Scenario([A0, A1, A2, B0, B1, B2])
chsh_space = NSpb_space(chsh_scenario)

chsh_lhvm = Local_deterministic_model(chsh_space)

for assignm in Local_deterministic_assignments(chsh_scenario):
    print(assignm)

for beh in chsh_lhvm:
    print(beh)


