from bellpy import *

nameasy = '2 [00] - [12] - [21] - [11] + [22]'
chsh = Inequality(name=nameasy)
print(chsh.name)
# output: +2.0 (00) - (11) - (12) + (22) 

chsh.update_qubit_violation(verbose=True)
chsh.update_qutrit_violation()
chsh.update_npa2()
chsh.update_npa3()

settings = chsh.behavior_space.nsettings
ns = NoSignaling(settings)
chsh.update_no_signaling_bound(ns)

print(chsh.summary())
chsh.save()
