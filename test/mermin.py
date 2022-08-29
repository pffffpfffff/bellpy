import bellpy as bell
import itertools as it

correlations = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2),
        (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2)]
B = bell.Custom_expectation_behavior_space(correlations)
lhv = bell.Local_deterministic_model(B)

bis = lhv.bell_inequalities()
print(bis)

