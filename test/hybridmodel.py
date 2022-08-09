import bellpy as bell

B = bell.Default_expectation_behavior_space(2,2,2)
lhv = bell.Local_deterministic_model(B, [[0,1], [2]])
