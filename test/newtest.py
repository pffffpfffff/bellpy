from bellpy import *
import time
import relabelings as rel

def test1():
    A = [Setting(0,i, [-1,1]) for i in [1,2]]
    B =[Setting(1,0,[1])] +  [Setting(1,i, [-1,1]) for i in [1]]
    C =[Setting(2,0,[1])] +  [Setting(2,i, [-1,1]) for i in [1]]

    sc = Scenario(A + B + C)

    ua = Unrestricted_assignments(sc)

    lda1 = Local_realistic_assignments(sc)

    hass = Hybrid_assignments(lda1, lda2, lda3)
    for a in hass:
        print(a)
    return 0

def test2():
    bs = Expectation_behavior_space([2,2])
    um = Unrestricted_model(bs)
    print(um)
    return 0

#test2()

def test3():
    bs = Default_expectation_behavior_space([2,2])
    print(bs.nsettings)
    print(bs.scenario.nsettings)
    ldm = Local_deterministic_model(bs)
    print(ldm)
    print(ldm.bell_inequalities())
    return 0

#test3()

def test4():
    
    A = [Setting(0,i, [-1,1]) for i in [1,2]]
    B =[Setting(1,0,[1])] +  [Setting(1,i, [-1,1]) for i in [1]]
    settings = [A[0], B[1]]
    jm = Joint_measurement(settings, mode = 'expect')
    for j in jm:
        print(j)

#test4()

def test5():
    bs = Custom_expectation_behavior_space([(0,0),(0,1), (1,0), (0,2), (2,0), (1,1), (2,1), (1,2), (2,2)])
#   print(bs.nsettings)
    print(bs.scenario.nsettings)
    ldm = Local_deterministic_model(bs)
    print(ldm)
   # print(ldm.bell_inequalities())
    cpt = Cone_projection_technique(ldm, true_facets=True)
    print(cpt.bell_inequalities())

    return 0

#test5()
def test6():
    """
    gen mp ine, 4p, 2m, 2o, only 3-1 partition
    """
    print('create model')
    correlations = list(it.product([1,2], repeat = 4))
    bs = Custom_expectation_behavior_space(correlations)
    lhv1 = Local_deterministic_model(bs, partition=[[0], [1,2,3]])
    lhv2 = Local_deterministic_model(bs, partition=[[1], [0,2,3]])
    lhv3 = Local_deterministic_model(bs, partition=[[2], [1,0,3]])
    lhv4 = Local_deterministic_model(bs, partition=[[3], [1,2,0]])
    print('model created')
    hybrid = Hybrid_model(lhv1, lhv2, lhv3, lhv4)
    print(lhv1)
    print(lhv1.data_frame().to_numpy())
    cpt = Cone_projection_technique(hybrid, true_facets=True)
    print('cpt created')
    cpt.add_party_symmetry()
#   cpt.add_only_full_body_correlations()
    bi = cpt.bell_inequalities()
    print('find bell inequalities')
    print(bi)
    return 0

#test6()
def test7():
    """
    Find inequalities for the Svetlichny scenario (with marginals)
    """
    correlations = [(0,0,0)] + list(it.product([1,2], repeat = 3))
    bs = Custom_expectation_behavior_space(correlations)
#   bs = Default_expectation_behavior_space([2,2,2])
#   print('labels', bs.labels)
    lhvm1 = Local_deterministic_model(bs,partition=[[0],[1,2]])
    print(lhvm1)
    lhvm2 = Local_deterministic_model(bs,partition=[[1],[0,2]])
    lhvm3 = Local_deterministic_model(bs,partition=[[2],[0,1]])
    hybrid = Hybrid_model(lhvm1, lhvm2, lhvm3)
    cpt = Cone_projection_technique(hybrid, true_facets=True)
    cpt.add_party_symmetry()
#   cpt.add_only_full_body_correlations()
    bis = cpt.bell_inequalities()
    print(bis)
#   bis = []
#   for i in range(20):
#       bis.append(cpt.random_bell_inequality())
#   ines = ineq.Inequalities(inequalities = bis)
#   print(ines)
    return 0

#test7()


def test8():
    """
    Find Svetlichny generalizations with 3 settings!
    """

    bs = Default_expectation_behavior_space(2,2,2)
    bs2 = Default_expectation_behavior_space(3,3,3)

    lhvm1 = Local_deterministic_model(bs,partition=[[0],[1,2]])
    lhvm2 = Local_deterministic_model(bs,partition=[[1],[0,2]])
    lhvm3 = Local_deterministic_model(bs,partition=[[2],[0,1]])
    hyb222 = Hybrid_model(lhvm1, lhvm2, lhvm3)

    lhvm4 = Local_deterministic_model(bs2 ,partition=[[0],[1,2]])
    lhvm5 = Local_deterministic_model(bs2 ,partition=[[1],[0,2]])
    lhvm6 = Local_deterministic_model(bs2 ,partition=[[2],[0,1]])
    hyb333 = Hybrid_model(lhvm4, lhvm5, lhvm6)

    svetl = Inequality(name = '+4.0 (000) - (111) - (112) + (122) + (222)') 
    ext_map = Canonic_extension_map(bs, (0,1,1), (1,0,1), (2,2,-1))
    ext_beh = Extended_behaviors(hyb222, svetl, ext_map)
    print(ext_beh)
    cpt = Cone_projection_technique(hyb333, true_facets = True)
    cpt.add_party_symmetry()
    cpt.add_extended_behaviors(ext_beh)
    print('cpt created')

   #bi = cpt.bell_inequalities(interactive=True)
    bis = []
    bis += cpt.random_bell_inequalities(n=100)
    
    ines = Inequalities(path='1502_ines')
    ines.load_inequalities_from_path()

#   ines.remove_duplicates(party = False, setting=True, outcome=True)
    ines.remove_duplicates()
    lines = [i.line for i in ines.inequalities]
    maxline = max(lines)
    for bi in ines:
        bi.line += maxline
    ines.inequalities += bis
#   ines.clear_all()
#   ines.save()
    print(ines, len(ines.inequalities))

   #print(len(bi))
    return 0
   
#test8()

def test9():
    """
    Find generalizations of chsh to i3322 scenario
    (yields i3322)
    """
    bs22 = Default_expectation_behavior_space(2,2)
    bs33 = Default_expectation_behavior_space(3,3)

    chsh = Inequality(name="2 (00) - (12) - (11) + (22)")
    chsh = Bell_inequality(bs22, bs22.canonic_array_to_array(chsh.table))

    print(chsh)
    lhvm = Local_deterministic_model(bs22)
    bis = lhvm.bell_inequalities()
    ext_map = Canonic_extension_map(bs22, (0,0,1), (1,0,1))
    em = Canonic_single_extension_map(bs22, 0)
    print('domain', ext_map.domain.scenario.nsettings)
    print('codomain', ext_map.codomain.scenario.nsettings)
    eb = Extended_behaviors(lhvm, chsh, ext_map)
  # print('nparray', eb.data_frame().to_numpy())
    print('eb', eb)
    lhvm3322 = Local_deterministic_model(bs33)
  ##print(lhvm.bell_inequalities())

    cpt = Cone_projection_technique(lhvm3322, true_facets=True)
    cpt.add_extended_behaviors(eb)
#   bi = cpt.bell_inequalities()
#   print(bi)

    bis = []
#   bis += cpt.random_bell_inequalities(n=100)
#   bis = cpt.bell_inequalities()
    cpt = load_cpt('facets')
    bis = cpt.run(cleanup = False)
    ines = Inequalities(inequalities = bis)
    print(ines)

test9()



