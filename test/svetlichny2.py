import bellpy as bell
import itertools as it

correlations = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2),
        (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2)]
B = bell.Custom_expectation_behavior_space(correlations)
lhv_c = bell.Local_deterministic_model(B, [[0,1], [2]])
lhv_b = bell.Local_deterministic_model(B, [[0,2], [1]])
lhv_a = bell.Local_deterministic_model(B, [[0], [1,2]])

svet = bell.Hybrid_model(lhv_c, lhv_b, lhv_a)
bis = svet.bell_inequalities()
print(bis)

