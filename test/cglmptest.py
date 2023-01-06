from bellpy import *

A0t = Setting(0, 0, [0])
A1t = Setting(0, 1, [0, 1, 2])
A2t = Setting(0, 2, [0, 1, 2])

B0t = Setting(1, 0, [0])
B1t = Setting(1, 1, [0, 1, 2])
B2t = Setting(1, 2, [0, 1, 2])

cglmp_scenario = Scenario([A0t, A1t, A2t, B0t, B1t, B2t])
cglmp_space = NSpb_space(cglmp_scenario)
cglmp_lhvm = Local_deterministic_model(cglmp_space)

coeffs = []

cglmp_dict = {
"A0B0: [0, 0]": 2,
"A1B1: [0, 0]": -1,
"A1B1: [0, 1]":  1, 
"A1B1: [1, 1]": -1,
"A1B1: [1, 2]":  1, 
"A1B1: [2, 0]":  1, 
"A1B1: [2, 2]": -1,
"A1B2: [0, 0]": -1,
"A1B2: [0, 2]":  1, 
"A1B2: [1, 0]":  1, 
"A1B2: [1, 1]": -1,
"A1B2: [2, 1]":  1, 
"A1B2: [2, 2]": -1,
"A2B1: [0, 0]": -1,
"A2B1: [0, 2]":  1, 
"A2B1: [1, 0]":  1, 
"A2B1: [1, 1]": -1,
"A2B1: [2, 1]":  1, 
"A2B1: [2, 2]": -1,
"A2B2: [0, 0]":  1, 
"A2B2: [0, 2]": -1, 
"A2B2: [1, 0]": -1,
"A2B2: [1, 1]":  1, 
"A2B2: [2, 1]": -1,
"A2B2: [2, 2]":  1  

}
v = np.zeros(cglmp_space.dimension)
for ev, coeff in cglmp_dict.items():
    v += coeff * cglmp_space.vector(ev)
    if coeff != 0:
        print(coeff)
        print("beh", Behavior_space_vector(cglmp_space, cglmp_space.vector(ev)))

cglmp = Bell_inequality(cglmp_space, v)
print("cglmp", cglmp)
print("is facet", cglmp_lhvm.is_facet(cglmp))


A0 = Setting(0, 0, [0])
A1 = Setting(0, 1, [0, 1])
A2 = Setting(0, 2, [0, 1])

B0 = Setting(1, 0, [0])
B1 = Setting(1, 1, [0, 1])
B2 = Setting(1, 2, [0, 1])

chsh_scenario = Scenario([A0, A1, A2, B0, B1, B2])
chsh_space = NSpb_space(chsh_scenario)
chsh = Bell_inequality(chsh_space,[1, 0, -1, 0, -1, 1, -1, 1, 1])

chsh_lhvm = Local_deterministic_model(chsh_space)

idd = lambda x: x
fun = lambda x: int(x>=1)

def fun2(x):
    if x == 0 or x == 2:
        return 0
    elif x == 1:
        return 1

# now define coarse grainings

cg0 = Coarse_graining(A0t, A0, idd) 
cg1 = Coarse_graining(A1t, A1, fun) 
cg2 = Coarse_graining(A2t, A2, fun) 
cg3 = Coarse_graining(B1t, B1, fun) 
cg4 = Coarse_graining(B2t, B2, fun) 
cg5 = Coarse_graining(B0t, B0, idd) 


inflation = Inflation(chsh_space, cglmp_space, cg0, cg1, cg2, cg3, cg4, cg5)
cglmpdef = inflation.deflate(cglmp)
cglmpdef.table = (cglmpdef.table/3).astype(int)
print("cglmp deflate", cglmpdef, chsh_lhvm.is_facet(cglmpdef))
print(cglmpdef.table)
#print("chsh         ", chsh, chsh_lhvm.is_facet(chsh))

bis = chsh_lhvm.bell_inequalities()
#bis.remove_duplicates()
print(bis)


# Check whether cglmp is saturated by extended behaviors
extension_maps = inflation.extension_maps()

problem_found = False
for em in extension_maps:
   # ebehaviors = Extended_behaviors(chsh_lhvm, cglmpdef, em)
    for beh in chsh_lhvm:
        if beh*cglmpdef == 0:
            eb = em(beh) 
            if eb*cglmp != 0:
                print("extension map", em)
                print('beh', beh, beh.table)
                print('deflated', inflation.coarse_grain(eb))
                print('full', beh.behavior_space.reconstruct_full(beh))
                print('eb', eb)
                print('eb full', eb.behavior_space.reconstruct_full(eb))
                print('eb 2',
eb.behavior_space.behavior_vector_from_full(eb.behavior_space.reconstruct_full(eb)))
                problem_found = True
                break
    if problem_found:
        break


"""
Coarse grain cglmp by hand
for 0 -> 0, 1 -> 1, 2 -> 1

"A0B0: [0, 0]": 2,
"A1B1: [0, 0]": -1,
"A1B1: [0, 1]":  1, 
"A1B1: [1, 0]":  1, 
"A1B1: [1, 1]": -1,
"A1B2: [0, 0]": -1,
"A1B2: [0, 1]":  1, 
"A1B2: [1, 0]":  1, 
"A1B2: [1, 1]": -1,
"A2B1: [0, 0]": -1,
"A2B1: [0, 1]":  1, 
"A2B1: [1, 0]":  1, 
"A2B1: [1, 1]": -1,
"A2B2: [0, 0]":  1, 
"A2B2: [0, 1]": -1, 
"A2B2: [1, 0]": -1,
"A2B2: [1, 1]":  1, 


"""
