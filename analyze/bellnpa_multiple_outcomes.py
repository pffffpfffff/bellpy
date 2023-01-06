# Explanation
# Represent the inequality 
# $$
# \sum_{k=1}^K c_k P(o_A^k, o_B^k, \ldots|i_A^k, i_B^k, \ldots)
# $$
# as a list
# $$
# \{[c_k, [ [o_A^k, o_B^k, \ldots], [i_A^k, i_B^k, \ldots]] ]\}_{k=1}^K,
# $$
# where $i_A = 0$ corresponds to the case that A's measurement is identity. 
###########################################################################
from bellpy import *
from ncpol2sdpa import generate_operators, SdpRelaxation, Probability

def npaBoundManual(prescenario,ineq,level,abq=0): # abq=0/1 with/without AB level # 0-th measurement is just identity
    scenario = prescenario # 0-th measurement is just identity
    le = len(scenario)
    P = Probability(*scenario)
    objective = sum(coe*P(*event) for coe,event in ineq)
    sdp = SdpRelaxation(P.get_all_operators())
    sdp.get_relaxation(level, objective=objective,
                           substitutions=P.substitutions,
                           extramonomials= [] if abq==0 else P.get_extra_monomials('AB')) # +AB level      
    sdp.solve()
    return sdp.primal
# minimal violation
def npaBound(BI,level,abq=0): # abq=0/1 with/without AB level
    evss = [list(zip(*[[st.value,st.label] for st in ev.settings])) for ev in BI.behavior_space.events]
    coes = BI.table
    ineq = list(zip(*[coes,evss])) # outcomes are [0,1,...]
    prescenario = [[len(A.outcomes) for A in sp] for sp in BI.behavior_space.scenario.settings_by_party()]
    res = npaBoundManual(prescenario,ineq,level,abq)
    return res
# example
level = 2
abq = 1
#npaBound(scenario,ineq,level,abq)
A0 = Setting(0, 0, [0])
A1 = Setting(0, 1, [0,  1])
A2 = Setting(0, 2, [0,  1])
A3 = Setting(0, 3, [0,  1])
B0 = Setting(1, 0, [0])
B1 = Setting(1, 1, [0,  1])
B2 = Setting(1, 2, [0,  1])
B3 = Setting(1, 3, [0,  1])

scenario = Scenario([A0, A1, A2, B0, B1, B2])
bs = NSpb_space(scenario)
ldm = Local_deterministic_model(bs)
bis = ldm.bell_inequalities()
print(bis)
bis.remove_duplicates()

print("unique bis", bis)

bds = [npaBound(bi,level,abq) for bi in bis]
print(bds)
