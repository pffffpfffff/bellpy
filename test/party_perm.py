from bellpy import *
import sympy.combinatorics as comb

s3 = comb.SymmetricGroup(3)
print(s3.generators)

A0 = Setting(0, 0, [0])
A1 = Setting(0, 1, [0, 1])
A2 = Setting(0, 2, [0, 1])
A3 = Setting(0, 3, [0, 1])
B0 = Setting(1, 0, [0])
B1 = Setting(1, 1, [0, 1])
B2 = Setting(1, 2, [0, 1])
B3 = Setting(1, 3, [0, 1])
C0 = Setting(2, 0, [0])
C1 = Setting(2, 1, [0, 1])
C2 = Setting(2, 2, [0, 1])
C3 = Setting(2, 3, [0, 1])


scenario = Scenario([A0, A1, A2, B0, B1, B2, C0, C1, C2])

bs = NSpb_space(scenario)

perms = [ Party_permutation(permutation = g, behavior_space = bs).matrix for g in s3.generators] 
for p in perms:
    print(p.astype(int))
    print(type(p))


def party_symmetry(behsp):
    G = SymmetricGroup(behsp.scenario.nparties)
    idd = np.eye(behsp.dimension, dtype = int)
    conds = [Party_permutation(permutation = g, behavior_space =
behsp).matrix.astype(int) \
             - idd for g in G.generators]
    return np.vstack(conds)

print(".......")
np.set_printoptions(threshold = np.inf)
par = party_symmetry(bs)
print(par)

     
