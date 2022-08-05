import numpy as np
from sympy.combinatorics import Permutation
import sympy as sym
import itertools as it
#import facets as fc


""" First some tools to deal with permutations """
def OnTuple(t, p): 
    rng = len(t)
    p = CompletePerm(p,rng)
    return [t[i^(~p)] for i in range(rng)]

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


def test1():
    l = [5,7,13,1]
    l2 = [2,1,0]
    p = Permutation([[0,1,2]])
    print(l2)
    print([x^p for x in l2])


    a = [5,8,7,13]
    b = [13,7,5,8]
    print(Permutation_from_image(a,b))
    print(Permutation_from_image(a,b).list())
    print(OnTuple(l, p))
    return 0



def CorrelationList(*settings):
    ranges = [range(s+1) for s in settings]
    return [c for c in it.product(*ranges)]

# print(CorrelationList(2,2,2))

def swap_parties(settings, i, j):
    C = CorrelationList(*settings)
    p = Permutation([[i, j]])
    C1 = [tuple(OnTuple(c, p)) for c in C]
    S = PermutationMat(Permutation_from_image(C, C1)) 
    return S


def party_symmetry(correlation_list):
    C = correlation_list
    G = np.zeros([0,len(C)])
    parties = len(C[0])
    for i in range(parties-1):
        p = Permutation([[i, i+1]])
        C1 = [tuple(OnTuple(c, p)) for c in C]
        G1 = PermutationMat(Permutation_from_image(C, C1)) - np.eye(len(C),dtype=int)
        G = np.vstack((G, G1))
    return G

def no_margs(correlation_list):
    C = correlation_list
    d = len(C)
    G = np.zeros([0,d])
    for i in range(1,d):
        if 0 in C[i]:
            g = np.zeros(d)
            g[i] = 1
            G = np.vstack((G, g))
    return G

nbody = lambda c: sum([1 for i in c if i>0])

def only_nbodymargs(C, n, mode="le"):
    """
    le: only margs with less or eq n parties allowed
    eq: only margs with exactly n parties allowed
    ge: only margs with more or eq n parties
    """
    d = len(C)
    G = np.zeros([0,d])
    if mode=="le":
        cond = lambda i: nbody(C[i])>n
    if mode=="ge":
        cond = lambda i: nbody(C[i])<n
    elif mode=="eq":
        cond = lambda i: nbody(C[i])!=n

    for i in range(1,d):
        if cond(i):
            g = np.zeros(d)
            g[i] = 1
            G = np.vstack((G, g))
    return G
        
