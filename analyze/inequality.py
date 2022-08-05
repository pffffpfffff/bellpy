import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import re
import itertools as it
import sys, os
import scipy.optimize as opt
import itertools as it

import bellpy.find.partysym as ps

import bellpy.analyze.nosignaling as ns
import bellpy.analyze.violationsobsnew as vl
import bellpy.analyze.violationobs as violationobs
import bellpy.analyze.violations
import bellpy.analyze.bellnp as bn
import bellpy.analyze.bellnpa as bellnpa

from bellpy.find.behavior import *
from bellpy.analyze.unitarynxn import mdot, unitarynxn
from bellpy.analyze.quantumbehavior import Quantum_behavior

class Inequality:
    """
    Bell inequality in QM
    """
    def __init__(self, ine: Bell_inequality = None, name = None):

        self.line = 0
        self.qubit_violation = None
        self.qubit_state = None
        self.qubit_obs = None
        self.qubit_settings = None

        self.qutrit_violation = None
        self.qutrit_state = None
        self.qutrit_obs = None
        self.qutrit_settings = None


        self.algebraic_bound = None
        self.classical_bound = None
        self.no_signaling_bound = None
        self.npa2 = None
        self.npa3 = None

        self.name = ''

        if ine is None:
            if name is None:
                raise ValueError('Provide at least one argument to Inequality')
            self.name = name
            self.update_table()
            self.behavior_space = Default_expectation_behavior_space(*[s-1 for s
                in np.shape(self.table)])
        else:
            self.table = ine.to_canonic_array()
            self.update_name()
            self.update_table() # to ensure that scenario is not unnecessarily large
            self.behavior_space = Default_expectation_behavior_space(*[s-1 for s
                in np.shape(self.table)])
            # behavior space can be smaller than the one of the bell_inequality
            # that generated the inequality, if it is a lifted version

        self.update_name()
        self.update_settings()
        self.update_classical_bound()
        self.update_algebraic_bound()
        self.various = {}


    def update_table(self):
        try:
            self.table = human2tab(self.name)
        except:
            raise Exception('Invalid name: ' + self.name)


    def update_name(self, round_coefficients=False):
        try:
            if round_coefficients:
                table = np.round(self.table, round_coefficients)
            else:
                table = self.table  
            self.name = tab2human(table)
        except:
            raise Exception('Invalid table: ', self.table)

    def get_line(self):
        return self.line


    def nice_name(self):
        tab = np.round(self.table).astype(int)
        return tab2human(tab)

    def update_settings(self):
        self.settings = list(np.array(np.shape(self.table)) - 1)
    def update_classical_bound(self):
        self.classical_bound = self.table.flatten()[0]
    def update_algebraic_bound(self):
        self.algebraic_bound = np.sum(np.absolute(self.table.flatten())) - self.classical_bound

    def update_no_signaling_bound(self, nosig):
        self.no_signaling_bound = nosig.ns_violation(self.table.flatten())[0]

    def update_qubit_violation(self, verbose=False, initvals=30, numtol=1e-6,
threshold=0, iobs = None, initrep=6):
        ret = False
        dims = [2]*len(np.shape(self.table))
        v, psi, obs = vl.violation(self.table, dims, repetitions=25,\
        initvals=initvals, initrep=initrep, numtol=numtol, verbose=verbose, iobs=iobs)
        nonecheck = (self.qubit_violation is None)
        isbetter = True
        if not nonecheck:
            isbetter = (v < self.qubit_violation + threshold)
        if isbetter:
            self.qubit_violation = v
            bstate, meas = beautifyket(psi, obs)
            self.qubit_obs = meas
            self.qubit_state = bstate
            self.qubit_settings = [[decompose(E) for E in party] for party in meas] 
            ret = True
        else:
            print('violation achieved:', v)
        return ret

    def update_qubit_violation_obsolete(self, verbose=False, initvals=10, numtol=1e-6,
threshold=0, iobs = None, initrep=6):
        ret = False
        dims = [2]*len(np.shape(self.table))
        v, rho, obs = violationobs.violation(self.table, dims, repetitions=25,\
        initvals=initvals, initrep=initrep, numtol=numtol, verbose=verbose, iobs=iobs)
        if v<self.qubit_violation + threshold:
            self.qubit_violation = v
            bstate, meas = beautifystate(psi, obs)
            self.qubit_obs = meas
            self.qubit_state = bstate
            self.qubit_settings = [[decompose(E) for E in party] for party in meas] 
            ret = True
        else:
            print('violation achieved:', v)
        return ret


    def update_qubit_violation_povm(self, verbose=False, initvals=10, numtol=1e-6,
threshold=0, initrep=6):
        ret = False
        dims = [2]*len(np.shape(self.table))
        v, rho, povms = violations.violation(self.table, dims, repetitions=15,
        initvals=initvals, initrep=initrep, numtol=numtol, verbose=verbose)
        obs = violations.povms2obs(povms)
        if (self.qubit_violation is None) or v < self.qubit_violation + threshold:
            self.qubit_violation = v
            bstate, meas = beautifystate(rho, obs)
            self.qubit_obs = meas
            self.qubit_state = bstate
            self.qubit_settings = [[decompose(eff) for eff in party] for party in meas]
            ret = True
        else:
            print('violation achieved:', v)
        return ret


    def update_qutrit_violation(self, verbose=False, initvals=20, numtol=1e-8,
threshold=None):
        ret = False
        if threshold is None:
            threshold = self.qutrit_violation
        dims = [3]*len(np.shape(self.table))
#           v, rho, povms = vl.violation(self.table, dims, repetitions=0,
#   initvals=initvals, initrep=6, numtol=numtol, verbose=verbose)
        v, psi, obs = vl.violation(self.table, dims, repetitions=20,
initvals=initvals, initrep=6, numtol=numtol, verbose=verbose)
      # obs = vl.povms2obs(povms)
        if threshold is None or v < threshold:
            self.qutrit_obs = obs
            self.qutrit_violation = v
            self.qutrit_state = psi
            self.qutrit_settings =[[decompose(eff) for eff in party] for party in obs]
            ret = True
        return ret

    def qubit_violation_state(self, psi, iobs=None, verbose = True):
        """ returns viol, obs """
        dims = [2]*len(np.shape(self.table))
        if np.size(psi) == np.prod(dims):
            return vl.violation_fix_state(self.table, dims, psi, iobs=iobs, verbose=verbose)
        else:
            print("psi has wrong dimension")
            return 0

    def qubit_violation_mixed_state(self, rho, iobs=None, verbose = True):
        """ returns viol, obs """
        d = np.shape(rho)[0]
        nparties = len(np.shape(self.table))
        ldim = int(d**(1/nparties))
        dims = [ldim]*len(np.shape(self.table))
        d = np.prod(dims)
        if np.shape(rho) == (d, d):
            return violationobs.violation_fix_state(self.table, dims, rho, iobs=iobs, verbose=verbose)
        else:
            print("psi has wrong dimension")
            return 0


    def qutrit_violation_state(self, psi, iobs=None, verbose = True):
        """ returns viol, povms """
        dims = [3]*len(np.shape(self.table))
        if np.size(psi) == np.prod(dims):
            return vl.violation_fix_state(self.table, dims, psi, iobs=iobs, verbose=verbose)
        else:
            print("psi has wrong dimension")
            return 0

    def update_npa2(self, verbose=False, checkfirst=True):
        if checkfirst:
            if self.npa2 is None:
                self.npa2 = bellnpa.npa(self.table, level = 2)
        else:
            self.npa2 = bellnpa.npa(self.table, level = 2)

    def update_npa3(self, verbose=False, checkfirst=True):
        print('updating npa3 value')
        if checkfirst:
            if self.npa3 is None:
                self.npa3 = bellnpa.npa(self.table, level = 3)
        else:
            self.npa3 = bellnpa.npa(self.table, level = 3)

    def percentual_margins(self, rnd=2, npa3=True):
        m23 = np.round(self.percentual_margin_qubit_qutrit(), rnd)
        mn =np.round(self.percentual_margin_qutrit_npa(npa3), rnd)
        ma =np.round(self.percentual_margin_class_alg(), rnd)
        mq =np.round(self.percentual_margin_class_quantum(), rnd)
        return {'23': m23, 'npa':mn, 'a': ma, 'q':mq}

    def percentual_margin_qubit_qutrit(self):
        return ((self.classical_bound -
self.qutrit_violation)/(self.classical_bound - self.qubit_violation) - 1)*100
    def percentual_margin_qutrit_npa(self, npa3=True):
        if npa3:
            print('Using npa3 value for npa')
            npaval = self.npa3
        else:
            print('Using npa2 value for npa')
            npaval = self.npa2
        return ((self.classical_bound - npaval)/(self.classical_bound -
self.qutrit_violation) - 1)*100
    def percentual_margin_class_alg(self):
        return (self.algebraic_bound/self.classical_bound - 1)*100
    def percentual_margin_class_quantum(self):
        return ((self.classical_bound -
self.qutrit_violation)/(self.classical_bound) - 1)*100

    def save(self, path = None, name=None):
        if name is None:
            name = str(self.line)
        if path is None:
            with open(name + ".ineq", "wb") as f:
                pickle.dump(self, f)
                f.close()
        else:
            with open(path + "/" + name + ".ineq", "wb") as f:
                pickle.dump(self, f)
                f.close()
    def save_dict(self, path=".", name=None):
        # save data as pickled dictionary
        dct = vars(self)
       #dct = {}
       #dct["type"] = self.type
       #dct["table"] = self.table
       #dct["line"] = self.line
       #dct["qubit_violation"] = self.qubit_violation
       #dct["qutrit_violation"] = self.qutrit_violation
       #dct["qubit_state"] = self.qubit_state
       #dct["qubit_settings"] = self.qubit_settings
       #dct["qutrit_state"] = self.qutrit_state
       #dct["qutrit_settings"] = self.qutrit_settings
       #dct["no_signaling_bound"] = self.no_signaling_bound
       #dct["npa2"] = self.npa2
       #dct["npa3"] = self.npa3
       #dct["qubit_obs"] = self.qubit_obs
       #dct["qutrit_obs"] = self.qutrit_obs
        if name is None:
            name = "{:05}".format(self.line)
        if not os.path.exists(path):
            os.mkdir(path)
        with open(path + "/" + name + ".ineqdct", "wb") as f:
            pickle.dump(dct,f)
            f.close()

    def __eq__(self, other):
    #   print("not ready")
        comps = [equal(self.name, self.name),\
        equal(self.line, other.line),\
        equal(self.settings, other.settings),\
        equal(self.table, other.table),\
        equal(self.qubit_violation, other.qubit_violation),\
        equal(self.qubit_state, other.qubit_state),\
        equal(self.qubit_settings, other.qubit_settings),\
        equal(self.qubit_obs, other.qubit_obs),\
        equal(self.qutrit_violation, other.qutrit_violation),\
        equal(self.qutrit_state, other.qutrit_state),\
        equal(self.qutrit_settings, other.qutrit_settings),\
        equal(self.qutrit_obs, other.qutrit_obs),\
        equal(self.no_signaling_bound, other.no_signaling_bound),\
        equal(self.algebraic_bound, other.algebraic_bound),\
        equal(self.classical_bound, other.classical_bound),\
        equal(self.npa2, other.npa2),\
        equal(self.npa3, other.npa3),\
        equal(self.type, other.type)]
        if all(comps):
            return True
        else:
           #print(comps)
            return False

    def __neq__(self, other):
        return not __eq__(self,other)

    def __call__(self, beh: Quantum_behavior):
        return np.dot(self.table.flatten(), beh.table)

    def summary(self):
        alph = ["A", "B", "C", "D", "E", "F", "G", "H"]
        S = ""
        lib = [list(it.product([alph[k]], list(range(1,self.settings[k]+1)))) for k in range(len(self.settings))]
        lib = [[x[0] + "{}".format(x[1]) for x in party] for party in lib]
        S += "\n\n--------------------------------------\n"
        S += "Number {} ------------------------------\n".format(self.line)
        S += self.name + " > 0"
        S += "\nno signaling:     {}".format(self.no_signaling_bound)
        if self.npa2!=100:
            S += "\nnpa2:             {}".format(self.npa2)
        if self.npa3!=100:
            S += "\nnpa3:             {}".format(self.npa3)
        S += """
qubit violation:  {}
qutrit violation: {}
algebraic bound: {}

Qubit violation
***************

violation: {}
state:     {}
""".format(self.qubit_violation, self.qutrit_violation,
    self.algebraic_bound, self.qubit_violation, np.round(self.qubit_state, 5))
        def rep_settings(arr):
            nonlocal S, lib
            k = 0
            for p in lib:
                j = 0
                for m in p: 
                    try:
                        S += m + ": {} \n".format(arr[k][j])
                    except:
                        continue
                    j = j + 1
                k = k + 1
            return 0
        rep_settings(self.qubit_settings)

        return S

    def to_tex(self, lbreak=10, round_coefficients = True):
      # if round_coefficients:
      #     table = np.round(self.table).astype(int)
      # else:
      #     table = self.table
        return bn.tabletotex(self.table, lbreak = lbreak, ints=round_coefficients)

    def __str__(self):
        return self.name

    def flatten(self):
        return self.table.flatten()
        
def equal(a, b):
    try:
        if type(a) != type(b):  
            return False
        elif type(a) == list:
            if len(a)!=len(b):
                return False
            else:
               #print([equal(a[k], b[k]) for k in range(len(a))])
                return all([equal(a[k], b[k]) for k in range(len(a))])
        elif type(a) == np.ndarray:
          # print('array found')
            if not np.all(np.shape(a)==np.shape(b)):
                return False
            else:
                return np.all(a==b)
        else:
            return a==b
    except:
        print('a', a)
        print('b', b)
        raise
        return 0
            

def test_equal():
    a = np.array([0,1,2])
    b = a.copy()
    aa = [a, a]
    bb = [b, b]

    print(equal(a,b))
    return 0


def load_ineq_dict(name, path = None):
    if path is None:
        with open(name + ".ineqdct", "rb") as f:
            dct = pickle.load(f)
            f.close()
    else:
        with open(path + "/" + name + ".ineqdct", "rb") as f:
            dct = pickle.load(f)
            f.close()
    return dct_to_ineq(dct)
            
    

####def dct_to_ineq(dct):
####    I = 0
####    print('dct', dct)
####    if dct["type"] == "Inequality":
####        I = Inequality(name=dct["name"])
####    elif dct["type"] == "Symmetric3PartyInequality":
####        I = Symmetric3PartyInequality(name=dct["name"])
####    for key, value in dct.items():
####        I.__dict__[key] = value
####    return I
def dct_to_ineq(dct):
    I = 0
   #print('dct', dct)
    I = Inequality(name=dct["name"])
    for key, value in dct.items():
        I.__dict__[key] = value
    return I
   


class Symmetric3PartyInequality(Inequality):
    # deprecated
    def __init__(self, name=None, table=None):
        super().__init__(name, table)
        self.type = "Symmetric3PartyInequality"

    def update_table(self):
        if self.table is None:
            try:
                self.table = human_sym2tab(self.name)
            except:
                print('Invalid name')
    def update_name(self, round_coefficients=False):
        if len(np.shape(self.table)) == 3:
            try:
                if round_coefficients:
                    table = np.round(self.table).astype(int)
                else:
                    table = self.table  
                # to do: check symmetry of table
                self.name = tab2human_sym(table)
            except:
                print('Invalid table')

    def to_tex(self, lbreak=10, round_coefficients=True):
        return tab2human_symtex(self.table, ints=round_coefficients)

class Inequalities:
    def __init__(self, inequalities = None, path="."):
        self.inequalities = inequalities
        self.add_inequalities(inequalities)
        self.path = path 

    def add_inequalities(self, inequalities):
        if isinstance(inequalities,bellpy.find.behavior.Bell_inequality_collection):
            inequalities = inequalities.bis
        self.inequalities = inequalities or []
        self.inequalities = [ Inequality(i) if type(i) is Bell_inequality else i
                for i in self.inequalities ]
        line = 0
        for i in self.inequalities:
            i.line = line
            line +=1

        return 0



    def set_path(self, path):
        self.path = path

    def __iter__(self):
        return iter(self.inequalities)

    def __getitem__(self, i):
        return self.inequalities[i] 

    def load_inequalities_from_path(self, path=None, dct = True):
        # dct = True --> load ineqdct (dictionaries) files
        # dct = False --> load ineq (Inequality) files
        if path is None:
            path = self.path
        for f in sorted(os.listdir(path)):
            if bool(re.search(r".*\.ineqdct$", f)) and dct:
                with open(path + "/" + f, "rb") as fl:
                    d = pickle.load(fl)
                    self.inequalities.append(dct_to_ineq(d))
                    fl.close()
            elif bool(re.search(r".*\.ineq$", f)) and not(dct):
                with open(path + "/" + f, "rb") as fl:
                    d = pickle.load(fl)
                    self.inequalities.append(d)
                    fl.close()

    def remove_duplicates(self, party = False, setting = False, outcome = False):
        behsp = self.inequalities[0].behavior_space
        bis = [ Bell_inequality(behsp,
            behsp.canonic_array_to_array(b.table)) for b in
            self.inequalities]
        bicoll = Bell_inequality_collection(bis)
        inds = bicoll.remove_duplicates(party, setting, outcome)

        new_ines = [ self.inequalities[i] for i in inds ]
        self.inequalities = new_ines
        for i in range(len(self.inequalities)):
            self.inequalities[i].line = i
       #self.clear_all()
       #self.save()
        return 0


    def clear_all(self):
        for f in os.listdir(self.path):
            if f.endswith(".ineqdct"):
                print(f)
                if self.path.endswith("/"):
                    os.remove(self.path+f)
                else:
                    os.remove(self.path+"/"+f)
        return 0

    def load_inequalities_from_olddcts(self, ttype, path):
        for f in sorted(os.listdir(path)):
            if bool(re.search(r".*\.ineq$", f)):
                with open(path + "/" + f, "rb") as fl:
                    d = pickle.load(fl)
                    inequ = old_style_dct_to_ineq(d, ttype)
                    if inequ is None:
                        print("Failure for file")
                        print(f)
                        break
                    self.inequalities.append(inequ)
                    fl.close()
            
    def add_inequalities_from_file(self, settings, ifile, iformat=None, path=None):
        """
        iformat is one of the following:
            - sym: for human_sym format
            - vecsym: for coefficient vector of sym 3 party ineq
            - name: If name is given in ordinary format
            - vec: coeffcient vector
        """
        if path is None:
            path = self.path
        with open(path + "/" + ifile, "r") as fl:
            line = fl.readline()
            linenum = 0
            while line:
                linenum += 1
                tab = 0
                if iformat=="vec":
                    line = line.split()
                    N = np.array([int(i) for i in line])
                    tab = np.reshape(N,list(settings + 1))
                    B = Inequality(table = tab)
                    B.set_line(linenum)
                    self.inequalities.append(B)
                elif iformat=="name":
                    tab = bn.string2table(line)
                    B = Inequality(table = tab)
                    B.set_line(linenum)
                    self.inequalities.append(B)
                elif iformat=="sym":
                    tab = human_sym2tab(line)
                    B = Symmetric3PartyInequality(table = tab)
                    B.set_line(linenum)
                    self.inequalities.append(B)
                elif iformat=="vecsym":
                    line = line.split()
                    N = np.array([int(i) for i in line])
                    tab = np.reshape(N,list(settings + 1))
                    B = Symmetric3PartyInequality(table = tab)
                    B.set_line(linenum)
                    self.inequalities.append(B)
                else:
                    print("Invalid iformat: choose sym, vecsym, name or vec")
                line = fl.readline()
            fl.close()
        return 0
            
    def print(self, lst=None):
        if lst is None:
            lst = list(range(0,len(self.inequalities)))
        S = ""
        for j in lst:
            i = self.inequalities[j]
            S += i.summary()
        print(S)
        return 0

    def __str__(self):
        S = ""
        for i in self.inequalities:
            S += i.name + "\n"
        return S

    def __len__(self):
        return len(self.inequalities)

    def save(self, path=None, dct=True, namefct= lambda i: "{:05}".format(Inequality.get_line(i))):
        if path is None:
            path = self.path
        if not os.path.exists(path):
            os.mkdir(path)
        if dct:
            for i in self.inequalities:
                i.save_dict(path=path, name=namefct(i)) 
        else:
            for i in self.inequalities:
                i.save(path=path, name=namefct(i)) 
        
    def compile_list(self, path=None, name="list", round_coefficients = True, inname="", func=Inequality.summary):
        if round_coefficients:
            for i in self.inequalities:
                i.update_name(round_coefficients)
        S = ""
        if path is None:
            path = self.path
        for i in self.inequalities:
            S += func(i)
        with open(path + "/" + name, "w") as tfile:
            tfile.write(S)
            tfile.close()
        return 0
    
    def coefficients_list(self, path=None, name="coefficients_list"):
        self.compile_list(name=name, func=lambda i: ' '.join(str(x) for x in i.table.flatten()) +"\n")
        return 0
       
    def name_list(self, path=None, name="name_list"):
        self.compile_list(name=name, func=lambda i: i.name +"\n")
        return 0
      

    def plot(self, path = None, name="", npa3=False, width=0.8):
        # make bar plot with class bound, qviol. and npa
        # plot npa3 value if npa3=True
        if path is None:
            path = self.path
        lnpa3 = []
        lnpa2 = []
        lqub = []
        lqut = []
        lclass = []
        llines = []
        for i in self.inequalities:
            lqub.append(i.classical_bound - i.qubit_violation)
            lqut.append(i.classical_bound - i.qutrit_violation)
            lnpa3.append(i.classical_bound - i.npa3)
            lnpa2.append(i.classical_bound - i.npa2)
            lclass.append(i.classical_bound)
            llines.append(i.line)
        plt.xlabel("Inequality number")
        plt.ylabel("violation")
        if npa3:
            plt.ylim(0,np.max(lnpa3)+1)
        else:
            plt.ylim(0,np.max(lnpa2) +1)
        plt.xlim(np.min(llines)-1,np.max(llines)+1)
        plt.title("Violations of " + name + " Generalizations")
        mgray = (0.7,0.7,0.7)
        mred = (0.7,0,0.0)
        mblue = (0,0,0.6)
        mgreen = (0.2,0.6,0)
        mcl = mcolors.TABLEAU_COLORS
        if npa3:
            plt.bar(range(np.min(llines),np.max(llines)+1), lnpa3,
color=mred,label="NPA bound",width=width)
        else:
            plt.bar(range(np.min(llines),np.max(llines)+1), lnpa2, color=mred,label="NPA bound",width=width)
        plt.bar(range(np.min(llines),np.max(llines)+1), lqut, color=mgray,label="Qutrit violation",width=width)
        plt.bar(range(np.min(llines),np.max(llines)+1), lqub, color=mgreen, label="Qubit violation",width=width)
        plt.bar(range(np.min(llines),np.max(llines)+1), lclass, color=mblue, label="Classical bound",width=width)
        plt.legend()
        plt.savefig(path +  '/fig'+name+'barplt.pdf')
       #plt.show()
        return 0

    def margin_table(self, npa3=True, lst = None):
        if lst is None:
            lst = list(range(0,len(self.inequalities)))
        S = """
\\begin{table}[h]
\\begin{tabular}{ll|llll}
Eq. & Number & $m_Q$ & $m_{32}$ & $m_N$ & $m_A$ \\\\\hline
"""
        for j in lst:
            i = self.inequalities[j]
            m = i.percentual_margins(npa3=npa3)
            S += """{} & {} & {} & {} & {} & {} \\\\
""".format(0, i.line, m["q"], m["23"], m["npa"], m["a"])
        S = S + """\end{tabular}
\end{table}"""
        print(S)
        return 0

    def select_by_function(self, f, number=3):
        flist = [f(i) for i in self.inequalities]
        n = np.minimum(number, len(self.inequalities))
        ind = np.argsort(flist)[::-1][0:n]
        return ind
        

            
    


def human_sym2tab(symf):
    s = re.sub(' ','',symf)
    s = re.sub('\+',' +',s)
    s = re.sub('-',' -',s)
    s = re.sub('\(',',(',s)
    s = s.split()
    s = [x.split(',') for x in s]
   #print(s)
    coeffs = [float(x.pop(0)) for x in s]
    s = [x[0] for x in s]
    s = [ re.sub(r'\(|\)',r'',x) for x in s ]
    s = [ re.sub(r'(\d)(\d)(\d)',r'\1,\2,\3',x) for x in s]
    s = [ x.split(',') for x in s]
    s = [[int(x) for x in y] for y in s]
    d = np.max(np.array(s)) + 1
    s = [tuple(x) for x in s]

    tab = np.zeros([d,d,d])

    for i in range(len(s)):
        for p in it.permutations(s[i]):
            tab[p] = coeffs[i]

    return tab


def tab2human_sym(B):
    meas = np.shape(B)[0]
    s = ''
    for i in range(meas):
        for j in range(i+1):
            for k in range(j+1):
                if B[i,j,k]!=0:
                    s = s + " {:+} ({}{}{})".format(B[i,j,k],i,j,k)
    return s



def tab2human(B):
    s = np.array(np.shape(B)) - 1
    parties = len(s)
    C = ps.CorrelationList(*s)
   #print('C', C)
    d = len(C)
    b = B.flatten()
   #print('b', b)
    st = ""
    blanksym = "(" + "{}"*parties + ")"
    blankasy = "[" + "{}"*parties + "]"
    if tuple(s) == tuple([s[0]]*parties):
       #print('first test', s)
        G = ps.party_symmetry(C)
        if np.all(np.dot(G, b)==0):
            # B is symmetric
           #print('symm')
            for i in range(d):
                if np.all(np.array(C[i]) == np.sort(C[i])) and b[i]!=0:
                    if b[i]!=1 and b[i]!=-1:
                        st += " {:+} ".format(b[i]) + blanksym.format(*C[i])
                    elif b[i]==1:
                        st += " + " + blanksym.format(*C[i])
                    elif b[i]==-1:
                        st += " - " + blanksym.format(*C[i])
                    else:
                        raise Exception("Error in tab2human")
            return st
    
   #print('unsymm')
    for i in range(d):
        if b[i]!=0:
            st += " {:+} ".format(b[i]) + blankasy.format(*C[i])
   #print('st', st)
    return st

def human2tab(s):
    sym = True if re.search("\(|\)", s) else False
    asym = True if re.search("\[|\]", s) else False
    consistent = (sym != asym)
    if not consistent:
        if (not sym and not asym):
            try:
                tab = bn.string2table(s)
                return tab
            except:
                raise Exception("Unknown format for Bell inequality")
        else:
            raise Exception("Don't mix [ and ( in inequality notation")
   #else:
       #print("parenthesis/bracket format detected")

    s = re.sub(' ','',s)
    s = re.sub('\+',' +',s)
    s = re.sub('-',' -',s)

    s = re.sub('\(',',(',s)
    s = re.sub('\[',',[',s)
    s = s.split()
    s = [x.split(',') for x in s]
#   print("hi", s)
    def to_float(x):
        if x=='+':
            return 1
        if x=='-':
            return -1
        return float(x)
    coeffs = [to_float(x.pop(0)) for x in s]
    s = [x[0] for x in s]
    s = [ re.sub(r'\(|\)',r'',x) for x in s ]
    s = [ re.sub(r'\[|\]',r'',x) for x in s ]
    s = [[ int(y) for y in tuple(x)] for x in s]
    parties = len(s[0])
#   print(s)
    s = np.array(s)
    dims = [ max(s[:,i]) + 1 for i in range(parties)]
    tab = np.zeros(dims) 
    ll = len(s)
    if asym:
       #print('asym')
        for i in range(ll):
            tab[tuple(s[i])] = coeffs[i]
    elif sym:
       #print('sym')
        dims = [max(dims)]*parties
        tab = np.zeros(dims) 
        for i in range(ll):
            for p in it.permutations(s[i]):
       #        print('p', p)
                tab[p] = coeffs[i]
    return tab
        
def testhumantab():
    B = np.random.random([3,3,3])
    name = tab2human(B)
    print(B)
    print(name)
    symname = re.sub("\[","(",name)
    symname = re.sub("\]",")",symname)
    tab = human2tab(symname)
    print(tab)
    return 0
 
#testhumantab()

def tab2human_symtex(iB, lbreak = 5, ints=True):
    meas = np.shape(iB)[0]
    if ints:
        B = np.round(iB).astype(int)
    else:
        B = iB
    s = '&'
    counter = 0
    for i in range(1, meas):
        for j in range(i+1):
            for k in range(j+1):
                if B[i,j,k]!=0:
                    if B[i,j,k]==1:
                        s = s + " - ({}{}{})".format(i,j,k)
                    elif B[i,j,k]==-1:
                        s = s + " + ({}{}{})".format(i,j,k)
                    else:
                        s = s + " {:+} ({}{}{})".format(-B[i,j,k],i,j,k)
                counter += 1
                if counter>lbreak:
                    s += """\\notag \\\\  &"""
                    counter = 0
    s += "\le {}".format(B[0,0,0])
    return s


""" ToDo:
-- merge 2 inequality objects if they refer to the same inequality
"""

def unify(I,J):
    for i in I.inequalities:
        for j in J.inequalities:
            if np.all(np.shape(i.table)== np.shape(j.table)):
                if np.all(i.table == j.table):
                    if   i.qubit_violation < j.qubit_violation:
                         j.qubit_violation = i.qubit_violation
                         j.qubit_state     = i.qubit_state
                         j.qubit_settings  = i.qubit_settings
                         j.qubit_obs       = i.qubit_obs
                    elif j.qubit_violation < i.qubit_violation:
                         i.qubit_violation = j.qubit_violation
                         i.qubit_state     = j.qubit_state
                         i.qubit_settings  = j.qubit_settings
                         i.qubit_obs       = j.qubit_obs

                    if   i.qutrit_violation < j.qutrit_violation:
                         j.qutrit_violation = i.qutrit_violation
                         j.qutrit_state     = i.qutrit_state
                         j.qutrit_settings  = i.qutrit_settings
                         j.qutrit_obs       = i.qutrit_obs
                    elif j.qutrit_violation < i.qutrit_violation:
                         i.qutrit_violation = j.qutrit_violation
                         i.qutrit_state     = j.qutrit_state
                         i.qutrit_settings  = j.qutrit_settings
                         i.qutrit_obs       = j.qutrit_obs
    return 0
              
def diff_inds(I, J, quick=False):
    inds = []
    if quick and len(I.inequalities)==len(J.inequalities):
        for k in range(len(I.inequalities)):
            i = I.inequalities[k]
            j = J.inequalities[k]
            if np.all(np.shape(i.table)== np.shape(j.table)):
                if np.all(i.table == j.table):
                    if i!=j:
                        inds.append((k, k))
    else:
        for k in range(len(I.inequalities)):
            i = I.inequalities[k]
            for l in range(len(J.inequalities)):
                j = J.inequalities[l]
                if np.all(np.shape(i.table)== np.shape(j.table)):
                    if np.all(i.table == j.table):
                        if i!=j:
                            inds.append((k, l))
    return inds

def diff(I, J, quick=False):
    inds = diff_inds(I, J, quick=quick)
    for ind in inds:
        print('I', I.inequalities[ind[0]].summary())
        print('J', J.inequalities[ind[1]].summary())
        pass
    print("Difference in following inequalities:", inds)
    return 0
                        
                    

idd = np.array([[1,0],[0,1]])
Sx = np.array([[0,1],[1,0]])
Sy = np.array([[0,-1j],[1j,0]])
Sz = np.array([[1,0],[0,-1]])

pauli = [idd, Sx, Sy, Sz]

def bloch(R):
    return 1/2*np.real(np.array([np.trace(np.dot(R,pauli[i])) for i in range(4)]))

def compbloch(c):
    if np.size(c)==3:
        return sum([c[i]*pauli[i+1] for i in range(3)])
    elif np.size(c)==4:
        return sum([c[i]*pauli[i] for i in range(4)])

def decompose(R):
    if np.shape(R)==(2,2):
        return bloch(R)
    else:
        return bn.decompose(R)

def compose(c):
    if np.size(c) == 4 or np.size(c)==3:
        return compbloch(c)
    else:
        return bn.compose(c)


ket2state = lambda psi: np.outer( psi, psi.conjugate())

def beautifystate(rho, obs):
    # restricted to qubits for now
    pur = np.trace(np.dot(rho,rho))
    if pur < 0.95:
        print('warning: optimal state is not pure')
   #print('pur', pur)
    lamda, V = np.linalg.eig(rho)
    ind = np.argmax(np.absolute(lamda))
    v = V[:, ind]
    return beautifyket(v, obs)

def beautifyket(v, obs):
    nparties = len(obs)
    dim = int(round(np.size(v)**(1/nparties)))
    npar = dim**2
    def locunpsi(par):
        u = [unitarynxn(dim, par[npar*i:npar*(i+1)]) for i in range(nparties)]
        sublists = [[2*i, 2*i + 1] for i in range(nparties)]
        sl2 = [2*i+1 for i in range(nparties)]
        args1 = list(it.chain(*list(zip(u,sublists))))
        psip = np.einsum(*args1, np.reshape(v, [dim]*nparties), sl2).flatten()
       #psip = np.einsum('ij,kl,mn,jln->ikm', *u, np.reshape(v, [2]*nparties)).flatten()
        return psip
    val = lambda par: np.sum(np.absolute(locunpsi(par)))
    S = opt.minimize(val, np.random.random(npar*nparties),bounds =
tuple([(-2*np.pi, 2*np.pi) for i in range(npar*nparties)]))
    bstate = locunpsi(S["x"])
    ind = np.argmax(np.absolute(bstate))
    phase = np.absolute(bstate[ind])/bstate[ind]
    bstate = bstate * phase
    def locunmeas():
        par = S["x"]
        u = [unitarynxn(dim, par[npar*i:npar*(i+1)]) for i in range(nparties)]
        applu = lambda x, u: np.einsum('ij,jk,kl -> il', u, x,
                u.conjugate().transpose())
        m = [[applu(x, u[i]) for x in obs[i]] for i in range(nparties)]
        return m
        
    meas = locunmeas()

    return bstate, meas


