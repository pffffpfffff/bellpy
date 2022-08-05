from bellpy.find.setting import *
from bellpy.find.assignments import *
from bellpy.find.jointmeasurement import *
from bellpy.find.probbehaviorspace import *
from bellpy.find.model import *
from bellpy.find.behavior import Default_expectation_behavior_space

A0 = Setting(0, 0, [1])
A1 = Setting(0, 1, [-1,  1])
A2 = Setting(0, 2, [-1,  1])
B0 = Setting(1, 0, [1])
B1 = Setting(1, 1, [-1,  1])
B2 = Setting(1, 2, [-1,  1])

#   jA1 = Joint_measurement([A1])
#   jA2 = Joint_measurement([A2])
#   jB1 = Joint_measurement([B1])
#   jB2 = Joint_measurement([B2])
#   jA1B1 = Joint_measurement([A1, B1])
#   jA1B2 = Joint_measurement([A1, B2])
#   jA2B1 = Joint_measurement([A2, B1])
#   jA2B2 = Joint_measurement([A2, B2])

#   jointmeas = [jA1, jA2, jB1, jB2, jA1B1, jA1B2, jA2B1, jA2B2]
scenario = Scenario([A0, A1, A2, B0, B1, B2])
jointmeas = scenario.joint_measurements()

def test1():

    beh_space = Probability_behavior_space(scenario)
    print(beh_space.labels)
    print(beh_space.dimension)

    #   a = Assignments(jointmeas)
    #   b = beh_space.behavior_from_assignments(a)
    #   print(b)
    #   print(b.table)
    #   print(len(b.table))

    ldm = Local_deterministic_model(beh_space)
    b1 = next(iter(ldm))
    print('behavior', b1.table)
    print('behavior', b1)
    print(len(b1.table))
    print(ldm.data_frame())
    return 0

def test2():
    global scenario
    bs = NSpb_space(scenario)
    print(bs.labels)
    print(bs.dimension)

    ldm = Local_deterministic_model(bs)
    lda = ldm.assignment_model
    for a in lda:
        print(a)
   #b1 = next(iter(ldm))
   #print('behavior', b1.table)
   #print('behavior', b1)
   #print(len(b1.table))
    print(ldm.data_frame())
    print('bis')
    print(ldm.bell_inequalities())
    return 0

#test2()

def test3():
    print('test 3')
    bs = Default_expectation_behavior_space(2,2,2)
    print(bs.labels)
    print(bs.dimension)

    ldm = Local_deterministic_model(bs)
    lda = ldm.assignment_model
    for a in lda:
        print(a)
   #b1 = next(iter(ldm))
   #print('behavior', b1.table)
   #print('behavior', b1)
   #print(len(b1.table))
    print(ldm.data_frame())
    print('bis')
    print(ldm.bell_inequalities())
    return 0

test3()

def test4():
    A0 = Setting(0, 0, [1])
    A1 = Setting(0, 1, [1,  2])# , 3])
    A2 = Setting(0, 2, [1,  2])# , 3])
    A3 = Setting(0, 3, [1,  2])# , 3])
    B0 = Setting(1, 0, [1])
    B1 = Setting(1, 1, [1,  2])# , 3])
    B2 = Setting(1, 2, [1,  2])# , 3])
    B3 = Setting(1, 3, [1,  2])# , 3])
    C0 = Setting(2, 0, [1])
    C1 = Setting(2, 1, [1,  2])# , 3])
    C2 = Setting(2, 2, [1,  2])# , 3])
    C3 = Setting(2, 3, [1,  2])# , 3])


    scenario = Scenario([A0, A1, A2, B0, B1, B2, C0, C1, C2])

    bs = NSpb_space(scenario)
    print(bs.labels)
    print(bs.dimension)
    ldm = Local_deterministic_model(bs)
    lda = ldm.assignment_model
    for a in lda:
        print(a)
   #b1 = next(iter(ldm))
   #print('behavior', b1.table)
   #print('behavior', b1)
   #print(len(b1.table))
    print(ldm.data_frame())
    print('bis')
    bis = ldm.bell_inequalities()
    print(bis)
    print(len(bis))

    return 0
#test4()


"""
Check whether all chsh ineqs are valid

implement relabelings

implement behavior_to_full_bsp function that completes behavior under
assumption of ns

act with rel on full beh space for simplicity
"""
