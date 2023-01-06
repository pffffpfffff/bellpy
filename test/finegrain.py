from bellpy import *

A0 = Setting(0, 0, [0])
A = Setting(0, 1, [0,1])

B0 = Setting(0, 0, [0])
B1 = Setting(0, 1, [0, 1, 2])
B2 = Setting(0, 2, [0, 1, 2])


S1 = Scenario([A0, A])
S2 = Scenario([A0, B1, B2])

bs_origin = NSpb_space(S1)
bs_target = NSpb_space(S2)

print(bs_origin.labels)
print(bs_target.labels)

fun = lambda x: int(x>=1)
cg1 = Coarse_graining(B1, A, fun) 
cg2 = Coarse_graining(B2, A, fun) 
cg3 = Coarse_graining(B0, A0, lambda x: x) 
inflation = Inflation(bs_origin, bs_target, cg1, cg2, cg3)

extensionmaps = inflation.extension_maps()

matrices = [em.matrix for em in extensionmaps]

print(bs_origin.dimension)
print(bs_target.dimension)


beh1 = Behavior(bs_origin, [1,2])
images = [str(em(beh1).table) for em in extensionmaps]

print(images)


