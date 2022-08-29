import copy
import dill
import inspect

class Function:
    def __init__(self, *functions, string = ''):
        self.fs = functions
        self.string = string

    def __call__(self, argument = None):
        x = argument
       #print('arg', argument)
        for f in self.fs:
            if x is None:
                x = f()
            else:
                x = f(x)
        return x

    def __matmul__(self, other: 'Function'):
        f = Function(*(other.fs + self.fs))
        f.string = ', '.join([other.string, self.string])
        return f
        
    def __pow__(self, n: int):
        return Function(*(self.fs*n))

    def __add__(self, other: 'Function'):
        f = Function( lambda x: self.__call__(x) + other.__call__(x)) 
        f.string = ' + '.join([self.string, other.string])
        return f
        

    def __radd__(self, other: int):
        # make sum() method work
        return self

#   def __neg__(self):
#       neg = Function(lambda x: -x)
#       return neg @ self

    def __sub__(self, other: 'Function'):
        f = Function( lambda x: self.__call__(x) - other.__call__(x)) 
        f.string = ' - ('.join([self.string, other.string]) + ')'
        return f

    def __mul__(self, other: 'Function'):
        f = Function( lambda x: self(x) * other(x))
        f.string = ' * ('.join([self.string, other.string]) + ')'
        return f

    def __str__(self):
       # return ','.join([dill.source.getsource(f) for f in self.fs])
        return self.string       
    
#   def __repr__(self):
#       return str(self)


class Function_by_string:
    def __init__(self, string):
        self.string = string

    def __call__(self, argument):
        return eval(self.string.format(argument))
       
    def __add__(self, other: 'Function'):
        f = Function_by_string(' + '.join([other.string, self.string]))
        return f
        
    def __radd__(self, other: int):
        # make sum() method work
        return self

    def __sub__(self, other: 'Function'):
        f = Function( lambda x: self.__call__(x) + other.__call__(x)) 
        f.string = ' - ('.join([other.string, self.string]) + ')'
        return f

    def __mul__(self, other: 'Function'):
        f = Function( lambda x: self(x) * other(x))
        f.string = '(' + ') * ('.join([other.string, self.string]) + ')'
        return f

    def __str__(self):
       # return ','.join([dill.source.getsource(f) for f in self.fs])
        return self.string       


class Dual_vector:
    def __init__(self, dualvec, string = ''):
        self.fs = dualvec
        self.string = string

    def __call__(self, argument = None):
        return np.dot(self.fs, argument)

    def __add__(self, other: 'Dual_vector'):
        f = Dual_vector(self.fs + other.fs)
        f.string = ' + '.join([self.string, other.string])
        return f
        

    def __radd__(self, other: int):
        # make sum() method work
        return self

    def __neg__(self):
        return Dual_vector(-self.vector, string = "-(" + self.string + ")")

    def __sub__(self, other: 'Dual_vector'):
        f = Dual_vector(self.fs - other.fs)
        f.string = ' - ('.join([self.string, other.string]) + ')'
        return f

    def __str__(self):
       # return ','.join([dill.source.getsource(f) for f in self.fs])
        return self.string       
     

import numpy as np

def test1():
    f1 = Function(np.sin)
    f2 = Function(lambda x: x**2)
    f3 = Function(np.exp)
    print(f3(f1(f2(5))))
    print((f3 @ f1 @ f2)(5))
    return 0

#test1()

def test2():
    f1 = Function(lambda x : x[0] + x[1])
    f2 = Function(lambda x : x**2)
    f3 = Function(lambda : 5)
   # print(f2(3))
    print(f3())
    print(f2(5))
    print(f1((3,5)))
    print((f2 @ f1)((3,5)))
    return 0

#test2()

def test3():

    f1 = Function(np.sin, string = 'sin')
    f2 = Function(np.cos, string = 'cos')
    print((f2 - f1)(3), f2(3) - f1(3))
    print((f2 * f1)(3), f2(3) * f1(3))
    l = [f1, f2]
    f3 = sum(l)
    print(f3(3), f1(3) + f2(3))
    print(f3)
    return 0

#test3()

