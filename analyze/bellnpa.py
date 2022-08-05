from ncpol2sdpa import generate_operators, SdpRelaxation
import itertools as it
import numpy as np


def perform_npa(BI, level=2, verbose=0):

    n_settings = list(np.array(np.shape(BI)) - 1)
    n_parties = len(n_settings)
    n_vars = sum(n_settings)


    X = generate_operators('X', n_vars, hermitian=True)
   #print('X',X)

    obs = []
    acc_sett = 0
    indexsublists = []
    for p in range(n_parties):
        obs.append([1] + X[acc_sett:acc_sett+n_settings[p]])
        indexsublists.append(list(range(acc_sett,acc_sett+n_settings[p])))
        acc_sett = acc_sett + n_settings[p]

    def samesublist(i,j):
        ssl = False
        for l in indexsublists:
            ssl = ssl or (i in l and j in l)
            if ssl:
                break
        return ssl

    def mult(meas):
        M = list(meas[:])
        def tms(M):
            M[0] = M.pop(0)*M[0]
            l = len(M)
            return l>1
        k = True
        while k:
            k = tms(M)
        return M[0]


    Corrs = [ mult(x) for x in it.product(*obs) ]
  # print(Corrs)
    Bifl = BI.flatten()
    obj = sum([Bifl[i]*Corrs[i] for i in range(len(Bifl))])

   #print(obj)

    substitutions1 = { X[i]*X[j]: X[j]*X[i] for i in range(n_vars) for j in
    range(i+1, n_vars) if not samesublist(i,j)}
    substitutions = { X[i]**2: 1 for i in range(n_vars) }
    substitutions.update(substitutions1)


    # Obtain SDP relaxation
    sdpRelaxation = SdpRelaxation(X,verbose=verbose)
    sdpRelaxation.get_relaxation(level, objective=obj,
                                 substitutions=substitutions)
    sdpRelaxation.solve(solver='mosek')
    ret = None
    if sdpRelaxation.status=="optimal":
        ret = sdpRelaxation.primal
    else:
        print("sdp.Relaxation.status is not optimal!")
    return ret

def npa(BI, level=2, verbose=0):
    ret = 0
    try:
        ret = perform_npa(BI, level, verbose)
    except:
        ret = None
        print('failed to compute level {} of npa hierarchy for current BI'.format(level))
    return ret 


def test():
   #bellfile = "all_ineq_3108.list"
    bellfile = "all_i3322_gen_wo_dup_sorted.list"
    B = b3.tab_from_human_sym_file(2,bellfile)
    print(b3.tab2human_sym(B))
   #bellfile = "all_hybrid1706_sorted.list"
   #B = bn.bitabfromline(400,bellfile)
   #print(B)

   #BI = b3.tab_from_human_sym_file(2,'all_i3322_gen_wo_dup_sorted.list')
    print(npa(B,2,verbose=0))
   #print(npa(B,3,verbose=1))
    return 0

#test()
