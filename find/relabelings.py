import sympy as sym
from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
import itertools as it
import numpy as np
import copy
import collections


class Representation:
    def __init__(self, *reps):
        self.reps = reps
        self.len = len(reps)
       #print(self.reps)
    def __call__(self, x):
        if self.len == 1:
            x = (x,)
           #print("converting to tuple")
        mats = []
       #print("x", x)
        for i in range(self.len):
           #print("i: ", i)
            try:
                m = self.reps[i](x[i])
            except:
                print("value of xi:", x[i])
                raise Exception("Error when calling representation")

            if m != 1:
                mats.append(m)
        return mdot_mats(*mats)
    def __add__(self, other):
        reps = self.reps + other.reps
        return Representation(*reps)
    def __radd__(self, other):
        if other==0:
            return self
        else:
            return self.__add__(other)

class Group:
    def __init__(self):
        self.list = iter([1])
        self.representation = Representation(lambda x: x)
        self.order = 1
        self.iterators = None

    def set_elements(self, gset, order=None):
        # gset should be an iterator
        tp = type(gset)
        if (tp is list) or (tp is set) or (tp is tuple):
            self.list = iter(gset)
        elif isinstance(gset, collections.Iterator):
            self.list = gset
        else:
            raise Exception("Invalid argument for set of group")

        self.iterators = [self.list]
        if order is None:
            self.order = self.get_order()
        else:
            self.order = order
        return 0

    def set_representation(self, rep):
        if type(rep) is Representation:
            self.representation = rep
        else:
            self.representation = Representation(rep)
        return 0 

    def set_iterators(self, iters: list, order=None):
        self.iterators = iters
        self.list = it.product(*[ copy.copy(itt) for itt in self.iterators ]) 
        if order is None:
            self.order = self.get_order()
        else:
            self.order = order

    def __iter__(self):
        return GroupIter(self)
    def __mul__(self, other):
        return mdot(self, other)
    def get_order(self):
        return len(list(copy.copy(self.list)))

    def __len__(self):
        return self.order

    def equivalent(self, vectors, labels=None):
        lvec = len(vectors)
        if labels is None:
            ind_to_label = {i:i for i in range(lvec)}
        else:
            ind_to_label = {i: labels[i] for i in range(lvec)}

        vecs_to_check = list(range(lvec)) # indices of vectors
        vec_dct = { labels[i]: vectors[i] for i in range(lvec) }
        tuples = [tuple(v) for v in vectors]
        while len(vecs_to_check)>1:
            counter = 0
            i = vecs_to_check.pop(0)
           #print("i", i)
            for g in self:
                gv = np.dot(g, vectors[i])
                tgv = tuple(gv)
                if counter%50==0:
                    print(counter)
                counter+=1
                for j in vecs_to_check:
                   #if np.all(gv == vectors[j]):
                    if tgv == tuples[j]:
                        ind_to_label[j] = ind_to_label[i]
                        vecs_to_check.remove(j)
                if vecs_to_check == []:
                    break
        classes = list(set(ind_to_label.values()))
        class_representants = [ vec_dct[i] for i in classes ]
        return class_representants, classes

    def equivalence_classes(self, vectors, labels = None, invariant = lambda v: tuple(np.sort(np.absolute(v)))):
        # This is the function to use, not the above, if possible
        # still much better: use the normal form functions below
        inv_vec_dct = {}
        inv_lab_dct = {}
        if labels is None:
            labels = list(range(len(vectors)))
        for i in labels:
            invv = invariant(vectors[i])
            if invv in inv_vec_dct:
                inv_vec_dct[invv].append(vectors[i])
                inv_lab_dct[invv].append(i)
            else:
                inv_vec_dct[invv] = [vectors[i]]
                inv_lab_dct[invv] = [i]
        class_representants_inds = []
        class_representants = []
        print("At least {} classes of BIs".format(len(inv_lab_dct)))
        for invv in inv_vec_dct:
            cr, ci = self.equivalent(inv_vec_dct[invv], inv_lab_dct[invv])
            class_representants += cr
            class_representants_inds += ci
        return class_representants, class_representants_inds

    def representants(self, objects):
        return self.equivalence_classes(self, [o.table for o in objects])[0]

       
        

def setting_outcome_normal_forms(bis, settG, outG):
    settnormforms = [Setting_normal_forms(b, settG) for b in bis]
    i = 1
    numbi = len(bis)
    print("finding setting normal forms")
    for s in settnormforms:
        s.find_good_versions()
        print(i, " out of ", numbi, len(s.versions), " versions")
        i += 1
    print("setting normal forms found")
    outnormforms = [Outcome_normal_forms(s.versions) for s in settnormforms]
    lout = len(outG)
    print(lout)
    i = 0
    for g in outG:
        i += 1
        for o in outnormforms:
            o.improve(g)
        print(i, "out of ", lout)
    normal_forms = [o.normal_form() for o in outnormforms]
    return normal_forms
   
def unique_inds_setting_outcome(bis, settG, outG):
    normal_forms = setting_outcome_normal_forms(bis, settG, outG)
    normal_forms = [tuple(f) for f in normal_forms]
    sn = set(normal_forms)
    print("set of normal forms", sn)
    print("normal forms", normal_forms)
    inds = [normal_forms.index(s) for s in sn]
    inds.sort()
    return inds
     

    
        

class Setting_normal_forms:
    def __init__(self, vec, group):
        self.versions = [vec] 
        self.vec = vec
        self.group = group

    def find_good_versions(self):
        for g in self.group:
            nv = np.dot(g, self.vec)
            srt = self.abs_lexsort_ge(nv, self.versions[0])
           #print("v0", self.versions[0])
           #print("nv", nv)
           #print("srt", srt)
            if srt == 1:
                self.versions = [nv]
            if srt == 0:
                self.versions.append(nv)
        return 0
            
    def abs_lexsort_ge(self, v1, v2):
        i=0
        ml = min(len(v1), len(v2))
        while np.abs(v1[i])==np.abs(v2[i]) and i< ml-1 :
            i += 1
        if i == ml-1:
            return 0
        else:
            if np.abs(v1[i])>np.abs(v2[i]):
                return 1
            else:
                return -1

class Outcome_normal_forms:
    def __init__(self, versions):
        self.versions = versions
        self.newversions = copy.copy(self.versions)
        self.len = len(self.newversions)

    def improve(self, relabeling):
        rv = [ np.dot(relabeling, v) for v in self.versions ]
        impr = [ rv[i] if self.order(self.newversions[i], rv[i]) == 1  \
else self.newversions[i] for i in range(self.len)]
        self.newversions = impr
        return 0

    def normal_form(self):
        n = self.newversions[0]
        for i in range(1, self.len):
            if self.order(self.newversions[i], n) == 1:
                n = self.newversions[i]
        return n

    def order(self, v1, v2):
        i = 0
        ml = min(len(v1), len(v2))
        while v1[i] == v2[i] and i<ml-1:
            i+= 1
        if i == ml:
            return 0
        else:
            if v1[i] > 0:
                return 1
            else:
                return -1 
                    
        
def mdot_mats(*mats):
    if len(mats)==0:
        return 1
    if len(mats)==1:
        return mats[0]
    args = []
    for i in range(len(mats)):
        args.append(mats[i]) 
        args.append([i, i+1])
    return np.einsum(*args)


def mdot(*groups):
   #rep = Representation(*[g.representation for g in groups])
   #rep = Representation(*(list(it.chain(*[g.representation.reps for g in groups]))))
    G = Group()

    if groups == ():
        return G

    rep = sum([g.representation for g in groups])
    neworder = np.prod([g.order for g in groups])
    iters = sum([g.iterators for g in groups], [])

    G.set_representation(rep)
    G.set_iterators(iters, order=neworder)
    return G        

class GroupIter:
    def __init__(self, group):
        self.items = copy.copy(group.list)
        self.group = group
    def __next__(self):
       #for item in self.items:
       #    yield self.group.representation(item)
        return self.group.representation(next(self.items))
        

def test1():
    A = np.random.random([3,3])
    B = np.random.random([3,3])
    C = np.random.random([3,3])

    g = Group()
    g.set_elements([A,B,C])

    for gg in g:
        print(gg)
        print("---")
    
    G = mdot(g,g,g,g,g,g,g,g)
    giter = iter(G)
    print("..........................")
    for i in range(1):
        print(next(giter)) 
    print("```````````````````")
    print(mdot_mats(A,A,A,A,A,A,A,A))
    return 0

#test1()

def test2():
    A = np.random.random([3,3])
    B = np.random.random([3,3])
    C = np.random.random([3,3])
    print(mdot_mats(A, B, C))
    return 0

class OutcomeRepresentation:
    def __init__(self, corr_list):
        self.parties = len(corr_list[1])
        self.settings = [max([c[p] for c in corr_list]) for p in range(self.parties)]
        self.tot_settings = sum(self.settings)
        self.ncorrs = len(corr_list)
        self.flips = []
        for p in range(self.parties):
            for s in range(1, self.settings[p]+1):
                self.flips.append(np.array([-1 if c[p]==s else 1 for c in corr_list], dtype=int)) 
    def __call__(self, x):
        vec = np.ones(self.ncorrs, dtype=int)
        isone = True
        for i in range(self.tot_settings):
            if x[i]:
                vec*=self.flips[i]
                isone = False
        if isone:
            return 1
        else:
            return np.diag(vec)
     
class OutcomeGroup(Group):
    """
    for outcomes -1, 1, consider all relabelings of outcomes
    given a list of correlations
    """
    def __init__(self, corr_list):
        self.orep = OutcomeRepresentation(corr_list)
        self.set_elements(it.product([0, 1], repeat=self.orep.tot_settings), order=2**self.orep.tot_settings)
        self.set_representation(self.orep)
        # 0 means no flip, 1 means flip outcome for this setting

""" First some tools to deal with permutations """
def OnTuple(t, p): 
    rng = len(t)
    p = CompletePerm(p,rng)
    return tuple([t[i^(~p)] for i in range(rng)])

def Permutation_from_image(preimage, image):
    l = [image.index(i) for i in preimage]
    p = CompletePerm(Permutation(l),len(image))
    return p

def CompletePerm(p,rng):
    pl = p.list()
    for r in range(rng):
        if not r in pl:
            p = [[r]]*p
    return p

def PermutationMat(p):
    l = p.list()
    size = len(l)
    M = np.eye(size, dtype=int)[l]
    return M

class SettingRepresentation:
    """
    Representation for permutation of settings of particular party
    """
    def __init__(self, corr_list, party):
        self.nsettings = max([ c[party] for c in corr_list])
        self.corr_list = corr_list
        self.ncorrs = len(corr_list)
        self.party = party
    def __call__(self, perm):
        if perm==Permutation(self.nsettings):
            return 1
        plist = [ perm(c[self.party]) for c in self.corr_list ]
        pcorr_list = []
        for i in range(self.ncorrs):
            newcorr = list(self.corr_list[i])
            newcorr[self.party] = plist[i]
            pcorr_list.append(tuple(newcorr))
       #print(perm)
        return PermutationMat(Permutation_from_image(self.corr_list, pcorr_list))


class SettingGroup(Group):
    """
    Permutation Group for settings of one party
    """
    def __init__(self, corr_list, party, store=True):
        rep = SettingRepresentation(corr_list, party)
        P = PermutationGroup(*[Permutation(i, i+1) for i in range(1, rep.nsettings)])
        self.order = len(P.elements)
        if store:
            self.set_elements(iter([rep(perm) for perm in P.elements]), order=self.order)
            self.set_representation(Representation(lambda x: x))
        else:
            self.set_representation(rep)
            self.set_elements(iter(P.elements), order=self.order)



class PartyGroup(Group):
    def __init__(self, corr_list):
        self.corr_list = corr_list
        self.parties = len(corr_list[1])
        self.perm_group = PermutationGroup(*[Permutation(i, i+1) for i in range(self.parties-1)])
        self.order = len(self.perm_group.elements)
        self.set_elements(iter([self.perm_to_matrix(perm) for perm in self.perm_group.elements]), order=self.order)
        self.set_representation(lambda x: x)
     
    def perm_to_matrix(self, perm):
        if perm==Permutation(self.parties-1):
            return 1
        pcorrlist = [OnTuple(c, perm) for c in self.corr_list]
        return PermutationMat(Permutation_from_image(self.corr_list, pcorrlist))
        
        
def test3():
    c = [(1,2,3), (2,3,1), (3,1,2), (1,3,2), (2,1,3), (3,2,1)]
    cc = [123, 231, 312, 132, 213, 321]
    pg = PartyGroup(c)
    for g in pg:
        print(g)
        print(np.dot(g, cc))
        print("...........")
    return 0

#test3()
        
def test4():
    c = [(1,2,3), (2,2,3), (3,2,3)]
    cc = [123, 223, 323]
    pg = SettingGroup(c, 0)
    for g in pg:
        print(g)
        print(np.dot(g, cc))
        print("...........")
    return 0

#test4()

def party_setting_outcome_relabelings(corr_list):
    s = [SettingGroup(corr_list, i) for i in range(len(corr_list[1]))]
    return mdot(PartyGroup(corr_list), *s, OutcomeGroup(corr_list))

def setting_outcome_relabelings(corr_list):
    s = [SettingGroup(corr_list, i) for i in range(len(corr_list[1]))]
    return mdot(*s, OutcomeGroup(corr_list))

def setting_relabelings(corr_list):
    s = [SettingGroup(corr_list, i) for i in range(len(corr_list[1]))]
    return mdot(*s)



    
def test5():
    c = [(0,0), (1,1), (1,2), (2,1), (2,2)]
    chsh = [2, -1, -1, -1, 1]
   #relgroup = PartyGroup(c)*SettingGroup(c,0)*SettingGroup(c,1)*OutcomeGroup(c)
    relgroup = setting_outcome_relabelings(c)
    notchsh = [3, 1,1,1,1]
    chshversions = [notchsh]
    for g in relgroup:
        chshversions.append(np.dot(g, chsh))
    print("order", relgroup.order)
    cr, ci = relgroup.equivalence_classes(chshversions)
    print("cr", cr)
    print("ci", ci)
    return 0


#test5()
def correlation_list_fb(*scen):
   #print('hi', list(scen))
    try:
       #print([s for s in scen])
        ranges = [ range(1, s+1) for s in scen ]
    except:
        print("fail")
    return [tuple([0]*len(scen))] + [ c for c in it.product(*ranges)]


def test6():
    c = correlation_list_fb(2,2,2)
    print(c)
    outG = OutcomeGroup(c)    
    print(len(outG))
    for g in outG:
        print(g)
    l = next(outG.list)
    print(l)
    outrep = outG.representation.reps[0]
    print(outrep(l))
    print("len", outG.representation.len)
    print(outG.representation(l))
    return 0

#test6()
def test7():
    c = [(0,0), (1,1), (1,2), (2,1), (2,2)]
    pG = PartyGroup(c)
    sG = mdot(*[SettingGroup(c, 0), SettingGroup(c,1)])
    oG = OutcomeGroup(c)
    G = mdot(pG, sG, oG)
    for g in G:
        print(g)
    return 0

#test7()
