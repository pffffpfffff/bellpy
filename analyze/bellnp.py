# gather some tools to deal with arbitrary 2 outcome BIs
# only works with python3, because list.copy()
import numpy as np
import re
import scipy.optimize as opt
import itertools as it
from itertools import chain, combinations
from functools import reduce

def randomunitary(n):
    M = (np.random.randn(n,n) + 1j*np.random.randn(n,n))/np.sqrt(2);
    q,r = np.linalg.qr(M);
    d = np.diagonal(r);
    d = d/np.absolute(d);
    q = np.multiply(q,d,q);
    return q

def dagger(v):
    return v.conjugate().transpose();

def ket(n,dim=2):
############ Example
#   |0 0 0> = (1 0 0 0 0 0 0 0)
#   |0 0 1> = (1 0) x (0 1 0 0) = (0 1 0 0 0 0 0 0)
#   |0 1 0> = (1 0) x (0 0 1 0) = (0 0 1 0 0 0 0 0)
#   |0 1 1> = (1 0) x (0 0 0 1) = (0 0 0 1 0 0 0 0)
#   |1 0 0> =                   = (0 0 0 0 1 0 0 0)

    parties = np.shape(n)
    if parties!=():
        parties = parties[0]
        #print(parties)
        v = np.zeros(dim**parties)
        numb = 0
        for i in range(0,parties):
            numb = numb + n[i]*dim**(parties - i - 1)
        v[numb] = 1
    else:
        v = np.zeros(dim)
        v[np.mod(n,dim)] = 1
    return v


def tabletohuman(array):
    alph = ["A","B","C","D","E","F","G","H"]
    shape = np.shape(array)
   #shsh = np.shape(shape)
    ranges = (range(x) for x in shape)
    
   #indices = lambda x, y: it.product(range(x), repeat=y)
    filterzeros = lambda x: it.filterfalse(lambda y: y[1]==0, x)
    comp_op = lambda y: map(lambda x: x[0] + "{}".format(x[1]),y)
    multiply = lambda x: reduce(lambda z, y: z + "*{}".format(y), x) if(x!=[]) else "idd"
    add = lambda z: reduce(lambda x,y: x + " " + y if y!="" else x,z)

   #thisindices = indices(shape[0],shsh[0])
    thisindices = it.product(*ranges)
    expr = map(lambda ind: ["{:+}".format(array.astype(int)[ind])] +\
list(comp_op(filterzeros(zip(alph,ind)))) if array[ind]!=0 else [""], thisindices)
    expr = map(lambda x: multiply(x), expr)
    expr = add(expr)
    return expr

def tabletotex(iarray, lbreak=20,ints=True, class_bound_mode=True):
    alph = ["A","B","C","D","E","F","G","H"]
    if ints:
        array = np.round(iarray).astype(int)
    else:
        array = iarray
    shape = np.shape(array)
    ranges = (range(x) for x in shape)
    tex = "&"
    counter = 0
 
    def ind2corr(ind):
        s = ""
        for i in range(np.shape(ind)[0]):
            if ind[i]!=0:
                s = s + alph[i] + "_{}".format(ind[i])
        return s

    for i in it.product(*ranges):
       #print(i)
        if array[i]!= 0:
            if class_bound_mode:
                if i == tuple([0]*len(shape)):
                    continue
                elif array[i] == 1:
                    tex = tex + "- \\langle " + ind2corr(i) + '\\rangle'
                elif array[i] == -1:
                    tex = tex + "+ \\langle " + ind2corr(i) + '\\rangle'
                else:
                    tex = tex + "{:+} \\langle ".format(-array[i]) + ind2corr(i) + '\\rangle'
            else:
                if array[i] == 1:
                    tex = tex + "+ \\langle " + ind2corr(i) + '\\rangle'
                elif array[i] == -1:
                    tex = tex + "- \\langle " + ind2corr(i) + '\\rangle'
                else:
                    tex = tex + "{:+} \\langle ".format(array[i]) + ind2corr(i) + '\\rangle'

            counter += 1
            if counter>=lbreak:
                tex = tex + """\\notag \\\\  &"""
                counter = 0

    tex += ("\le {}".format(array[tuple([0]*len(shape))]) if class_bound_mode
else "\ge 0")
   #print(tex)
    return tex

def string2table(inputstring, settings=None):
    # settings could be [2,3] for 2 (nontrivial) settings on A and 3 on B
    def string2index(stri):
        # "A1*C3" --> (103)
        index = [0 for n in range(numofparties)]
        if stri!='':
            s = stri.split('*')
            s = [re.sub(r'([A-Z])',r'\1,',x) for x in s] # A1 -> A,1 
            s = [x.split(',') for x in s] # ["A,1","B,1"] -> [["A","1"],["B","1"]]
            s = [[partydict[x[0]],int(x[1])] for x in s] # [["A",1],["B",1]] -> [[0,1],[1,1]]
            for x in s:
                index[x[0]] = x[1]
        return tuple(index)

    string = re.sub(' ','',inputstring)
    string = re.sub(r'([+-])([A-Z])',r'\1 1*\2',string)
    string = re.sub(r'^([A-Z])',r'1*\1',string)
    string = re.sub(' ','',string)
#   print(string)

    # find out how many parties are there
    parties = list(set(re.findall(r'[A-Z]',string)))
    parties.sort()
    numofparties = len(parties)
    partydict = { parties[i] : i for i in range(numofparties) }

    # separate terms
    string = re.sub(r'([+-])',r' \1',string)
    string = re.sub(r'^ *','',string)
    string = re.sub(r' *$','',string)
#   print(string)
    l = string.split(' ')    
#   print(l)
    # remove empty set in the beginning
#   l = l[1:]

    # separate coefficients and operators
    l = [re.sub('\*',',',x,1) for x in l] #only 1 substitution
    l = [x.split(',') for x in l]
    coeff = [int(x.pop(0)) for x in l]

    # now all the operators are in l 
    l = list(map(lambda x: x[0] if (x!=[]) else '',l))
    l = [string2index(x) for x in l]

    # now find number of measurements per party    
    if settings is None:
        larr = np.array(l).transpose()
        dims = [np.max(x)+1 for x in larr]
    else:
        dims = np.array(settings) + 1
    arr = np.zeros(dims)
    for i in range(len(coeff)):
        arr[l[i]] = coeff[i]

    return arr

def belloperator(B,settings):
    # first check sanity
    dims = [np.shape(x[0])[0] for x in settings]
    settwith1 = [list(s) for s in settings]
    for x in settwith1:
        x.insert(0,np.eye(*(np.shape(x[0]))))
    shB = np.shape(B)
#   print('settwith1',settwith1)
    num_of_ops = np.array([len(x) for x in settwith1])
    Bo = np.zeros([np.product(dims),np.product(dims)])
    if np.all(shB==num_of_ops):
        sys_index = list(range(len(dims)))
        index_ranges = tuple([range(x) for x in shB])
        for i in it.product(*index_ranges): 
            operator_indices = zip(sys_index,i)
            ops = [settwith1[x[0]][x[1]] for x in operator_indices]
            Bo = Bo + B[i]*mkronf(ops)
    return Bo
        
def bifromline(lnum, bellfile, settings):
    f = open(bellfile,'r')
    line = f.readlines()[lnum-1]
    line = line.split()
    N = np.array([int(i) for i in line])
   #print(N) 
    return np.reshape(N,list(np.array(settings, dtype=int) + 1))

def bitabfromline(lnum, bellfile):
    f = open(bellfile,'r')
    line = f.readlines()[lnum-1]
    return string2table(line)

def intbase(n,b,digits=2):
    # representation of integer in basis b.
    # returns array with the digits
    r = np.array([],dtype=int)
    nn = np.mod(n,b**digits)
    for i in range(1,digits):
        d = np.floor_divide(nn,b**(digits-i))
        m = np.remainder(nn,b**(digits-i))
        nn = nn - d*b**(digits-i)
        r = np.append(r,d)
    r = np.append(r,m)
    return r

def ket(n,dim=2):
############ Example
#   |0 0 0> = (1 0 0 0 0 0 0 0)
#   |0 0 1> = (1 0) x (0 1 0 0) = (0 1 0 0 0 0 0 0)
#   |0 1 0> = (1 0) x (0 0 1 0) = (0 0 1 0 0 0 0 0)
#   |0 1 1> = (1 0) x (0 0 0 1) = (0 0 0 1 0 0 0 0)
#   |1 0 0> =                   = (0 0 0 0 1 0 0 0)

    parties = np.shape(n)
    if parties!=():
        parties = parties[0]
        #print(parties)
        v = np.zeros(dim**parties)
        numb = 0
        for i in range(0,parties):
            numb = numb + n[i]*dim**(parties - i - 1)
        v[numb] = 1
    else:
        v = np.zeros(dim)
        v[np.mod(n,dim)] = 1
    return v

def braket(ket1,ket2):
    return np.dot(ket1.conjugate().transpose(),ket2)

def ketbra(ket1,ket2):
    return np.outer(ket1,ket2.conjugate().transpose())



def hmatbas(n,d):
    # defines a hermitian matrix basis, this is necessary because
    # the solver for the sdp will only accept real coefficients as
    # input, but density matrices are complex. By writing it in 
    # such a matrix, the coefficients become real
    a = intbase(n,d)
    if a[0]==a[1]:
        M = ketbra(ket(a[0],d),ket(a[0],d))
    elif a[0]<a[1]:
        M = (ketbra(ket(a[0],d),ket(a[1],d)) +ketbra(ket(a[1],d),ket(a[0],d)))/np.sqrt(2)
    else:
        M = 1j*(ketbra(ket(a[0],d),ket(a[1],d)) - ketbra(ket(a[1],d),ket(a[0],d)))/np.sqrt(2)
    return M


def decompose(M,basis=hmatbas):
    # returns M as a vector in the hmatbas basis
    s = np.size(M)
    d = int(np.sqrt(s))
    v = np.zeros(s,dtype=np.complex)
    for i in range(0,s):
        v[i] = np.trace(np.dot(M,basis(i,d)))
    return v

def compose(v,basis=hmatbas):
    # interprets v as matrix decomposition in the hmatbas basis
    # and reconverts it into the computational basis
    d = int(np.sqrt(np.size(v)))
    M = 0
    for i in range(0,np.size(v)):
       M = M + v[i]*basis(i,d)
    return M

           
        
        
    
