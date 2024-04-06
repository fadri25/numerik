import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import math

# Aufgabe 1a
x = sp.Symbol('x')
fx = x * sp.cos(2*x)
ffx = sp.diff(fx, x)
fffx = sp.diff(ffx,x)
ffffx = sp.diff(fffx,x)
print(ffx)
print(fffx)
print(ffffx)

# Aufgabe 1b
def f(x):
    return x * np.cos(2*x)
x0 = np.pi
def taylor_factory_n(x0, n):
    fk_list = [f(x0) for _ in range(n)]
    derivatives = np.array([fk for fk in fk_list])
    factorials = np.array([math.factorial(n) for n in range(len(fk_list))])
    coefficients = derivatives / factorials

    return lambda x: np.sum([ck * (x - x0)**k for k, ck in enumerate(coefficients)], axis=0)

tx = taylor_factory_n(x0, 3)
print(tx(x0))

# Aufgabe 1c
x0 = np.pi
x1 = np.pi + 0.1
t1 = taylor_factory_n(x0, 3)
app = abs(tx(x1))-abs(f(x1))
print(app)

# Aufgabe 2a
p1 = 0
p2 = 1
p3 = -3*x-12
p4 = np.log(1) - 2 * (x-0.5)
p5 = 4 - 2 * x
p6 = 1 + 2 * x
p7 = 8 * x - 4
p8 = 3 - x**2
p9 = taylor_factory_n(-4, 1)
print(p9(-4))

# Aufgabe 6a
