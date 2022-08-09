import bellpy as bell

B = bell.Default_expectation_behavior_space(3,3)
lhv = bell.Local_deterministic_model(B)

bis = lhv.bell_inequalities()
inds = bis.remove_duplicates(party=True, setting = True, outcome=True)
print(bis)
print(inds)
