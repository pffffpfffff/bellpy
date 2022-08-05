import copy

class Function:
    def __init__(self, *functions):
        self.fs = functions

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
        return Function(*(other.fs + self.fs))
        
    def __pow__(self, n: int):
        return Function(*(self.fs*n))

    def __add__(self, other: 'Function'):
        return Function( lambda x: self.__call__(x) + other.__call__(x)) 

    def __radd__(self, other: int):
        # make sum() method work
        return self

    def __neg__(self):
        neg = Function(lambda x: -x)
        return neg @ self

    def __sub__(self, other: 'Function'):
        return self + (-other)

    def __mul__(self, other: 'Function'):
        return Function( lambda x: self(x) * other(x))


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

    f1 = Function(np.sin)
    f2 = Function(np.cos)
    print((f2 - f1)(3), f2(3) - f1(3))
    print((f2 * f1)(3), f2(3) * f1(3))
    l = [f1, f2]
    f3 = sum(l)
    print(f3(3), f1(3) + f2(3))
    return 0

#test3()
