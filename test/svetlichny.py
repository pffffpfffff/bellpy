import bellpy as bell

B = bell.Default_expectation_behavior_space(2,2,2)
lhv_c = bell.Local_deterministic_model(B, [[0,1], [2]])
lhv_b = bell.Local_deterministic_model(B, [[0,2], [1]])
lhv_a = bell.Local_deterministic_model(B, [[0], [1,2]])

svet = bell.Hybrid_model(lhv_c, lhv_b, lhv_a)

