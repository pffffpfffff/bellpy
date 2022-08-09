import bellpy as bell

A1 = bell.Setting(0,1,[-1,1])
A2 = bell.Setting(0,2,[-1,1])
B1 = bell.Setting(1,1,[-1,1])
B2 = bell.Setting(1,2,[-1,1])

chsh_scenario = bell.Scenario([A1, A2, B1, B2])
print(chsh_scenario.nsettings)
